/*
Copyright 2019 The Vitess Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package encryptedreplication

import (
	"flag"
	"os"
	"os/exec"
	"path"
	"testing"

	"github.com/stretchr/testify/require"

	"vitess.io/vitess/go/constants/sidecar"
	"vitess.io/vitess/go/test/endtoend/cluster"
	"vitess.io/vitess/go/test/endtoend/encryption"
	"vitess.io/vitess/go/vt/log"
	"vitess.io/vitess/go/vt/utils"
)

var (
	clusterInstance *cluster.LocalProcessCluster
	keyspace        = "test_keyspace"
	hostname        = "localhost"
	shardName       = "0"
	cell            = "zone1"
	certDirectory   string
)

// This test makes sure that we can use SSL replication with Vitess
func TestSecure(t *testing.T) {
	testReplicationBase(t, true)
	testReplicationBase(t, false)
}

// This test makes sure that we can use SSL replication with Vitess.
func testReplicationBase(t *testing.T, isClientCertPassed bool) {
	flag.Parse()

	// initialize cluster
	_, err := initializeCluster(t)
	require.Nil(t, err, "setup failed")

	defer teardownCluster()

	primaryTablet := *clusterInstance.Keyspaces[0].Shards[0].Vttablets[0]
	replicaTablet := *clusterInstance.Keyspaces[0].Shards[0].Vttablets[1]

	if isClientCertPassed {
		replicaTablet.VttabletProcess.ExtraArgs = append(replicaTablet.VttabletProcess.ExtraArgs,
			utils.GetFlagVariantForTests("--db-flags"), "2048",
			utils.GetFlagVariantForTests("--db-ssl-ca"), path.Join(certDirectory, "ca-cert.pem"),
			utils.GetFlagVariantForTests("--db-ssl-cert"), path.Join(certDirectory, "client-cert.pem"),
			"--db-ssl-key", path.Join(certDirectory, "client-key.pem"),
		)
	}

	// start the tablets
	for _, tablet := range []cluster.Vttablet{primaryTablet, replicaTablet} {
		_ = tablet.VttabletProcess.Setup()
	}

	// Reparent using SSL (this will also check replication works)
	err = clusterInstance.VtctldClientProcess.InitializeShard(keyspace, shardName, clusterInstance.Cell, primaryTablet.TabletUID)
	if isClientCertPassed {
		require.NoError(t, err)
	} else {
		require.Error(t, err)
	}

	err = clusterInstance.StartVTOrc(keyspace)
	require.NoError(t, err)
}

func initializeCluster(t *testing.T) (int, error) {
	var mysqlProcesses []*exec.Cmd
	clusterInstance = cluster.NewCluster(cell, hostname)

	// Start topo server
	if err := clusterInstance.StartTopo(); err != nil {
		return 1, err
	}

	// create certs directory
	log.Info("Creating certificates")
	certDirectory = path.Join(clusterInstance.TmpDirectory, "certs")
	_ = encryption.CreateDirectory(certDirectory, 0700)

	err := encryption.ExecuteVttlstestCommand("CreateCA", "--root", certDirectory)
	require.NoError(t, err)

	err = encryption.ExecuteVttlstestCommand("CreateSignedCert", "--root", certDirectory, "--common-name", "Mysql Server", "--serial", "01", "server")
	require.NoError(t, err)

	err = encryption.ExecuteVttlstestCommand("CreateSignedCert", "--root", certDirectory, "--common-name", "Mysql Client", "--serial", "02", "client")
	require.NoError(t, err)

	extraMyCnf := path.Join(certDirectory, "secure.cnf")
	f, err := os.Create(extraMyCnf)
	require.NoError(t, err)

	_, err = f.WriteString("require_secure_transport=" + "true\n")
	require.NoError(t, err)
	_, err = f.WriteString("ssl-ca=" + certDirectory + "/ca-cert.pem\n")
	require.NoError(t, err)
	_, err = f.WriteString("ssl-cert=" + certDirectory + "/server-cert.pem\n")
	require.NoError(t, err)
	_, err = f.WriteString("ssl-key=" + certDirectory + "/server-key.pem\n")
	require.NoError(t, err)

	err = f.Close()
	require.NoError(t, err)

	err = os.Setenv("EXTRA_MY_CNF", extraMyCnf)

	require.NoError(t, err)

	for _, keyspaceStr := range []string{keyspace} {
		KeyspacePtr := &cluster.Keyspace{Name: keyspaceStr}
		keyspace := *KeyspacePtr
		if err := clusterInstance.VtctldClientProcess.CreateKeyspace(keyspace.Name, sidecar.DefaultName, ""); err != nil {
			return 1, err
		}
		shard := &cluster.Shard{
			Name: shardName,
		}
		for i := 0; i < 2; i++ {
			// instantiate vttablet object with reserved ports
			tabletUID := clusterInstance.GetAndReserveTabletUID()
			tablet := clusterInstance.NewVttabletInstance("replica", tabletUID, cell)

			// Start Mysqlctl process
			mysqlctlProcess, err := cluster.MysqlCtlProcessInstance(tablet.TabletUID, tablet.MySQLPort, clusterInstance.TmpDirectory)
			if err != nil {
				return 1, err
			}
			tablet.MysqlctlProcess = *mysqlctlProcess
			proc, err := tablet.MysqlctlProcess.StartProcess()
			if err != nil {
				return 1, err
			}
			mysqlProcesses = append(mysqlProcesses, proc)
			// start vttablet process
			tablet.VttabletProcess = cluster.VttabletProcessInstance(
				tablet.HTTPPort,
				tablet.GrpcPort,
				tablet.TabletUID,
				clusterInstance.Cell,
				shardName,
				keyspace.Name,
				clusterInstance.VtctldProcess.Port,
				tablet.Type,
				clusterInstance.TopoProcess.Port,
				clusterInstance.Hostname,
				clusterInstance.TmpDirectory,
				clusterInstance.VtTabletExtraArgs,
				clusterInstance.DefaultCharset)
			tablet.Alias = tablet.VttabletProcess.TabletPath
			shard.Vttablets = append(shard.Vttablets, tablet)
		}
		keyspace.Shards = append(keyspace.Shards, *shard)
		clusterInstance.Keyspaces = append(clusterInstance.Keyspaces, keyspace)
	}
	for _, proc := range mysqlProcesses {
		err := proc.Wait()
		if err != nil {
			return 1, err
		}
	}
	return 0, nil
}

func teardownCluster() {
	clusterInstance.Teardown()
}
