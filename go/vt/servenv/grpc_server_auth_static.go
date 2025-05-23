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

package servenv

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/spf13/pflag"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	"vitess.io/vitess/go/vt/log"
	"vitess.io/vitess/go/vt/utils"
)

var (
	credsFile string
	// StaticAuthPlugin implements AuthPlugin interface
	_ Authenticator = (*StaticAuthPlugin)(nil)
)

// The datatype for static auth Context keys
type staticAuthKey int

const (
	// Internal Context key for the authenticated username
	staticAuthUsername staticAuthKey = 0
)

func registerGRPCServerAuthStaticFlags(fs *pflag.FlagSet) {
	utils.SetFlagStringVar(fs, &credsFile, "grpc-auth-static-password-file", credsFile, "JSON File to read the users/passwords from.")
}

// StaticAuthConfigEntry is the container for server side credentials. Current implementation matches the
// the one from the client but this will change in the future as we hooked this pluging into ACL
// features.
type StaticAuthConfigEntry struct {
	Username string
	Password string
	// TODO (@rafael) Add authorization parameters
}

// StaticAuthPlugin  implements static username/password authentication for grpc. It contains an array of username/passwords
// that will be authorized to connect to the grpc server.
type StaticAuthPlugin struct {
	entries []StaticAuthConfigEntry
}

// Authenticate implements AuthPlugin interface. This method will be used inside a middleware in grpc_server to authenticate
// incoming requests.
func (sa *StaticAuthPlugin) Authenticate(ctx context.Context, fullMethod string) (context.Context, error) {
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		if len(md["username"]) == 0 || len(md["password"]) == 0 {
			return nil, status.Errorf(codes.Unauthenticated, "username and password must be provided")
		}
		username := md["username"][0]
		password := md["password"][0]
		for _, authEntry := range sa.entries {
			if username == authEntry.Username && password == authEntry.Password {
				return newStaticAuthContext(ctx, username), nil
			}
		}
		return nil, status.Errorf(codes.PermissionDenied, "auth failure: caller %q provided invalid credentials", username)
	}
	return nil, status.Errorf(codes.Unauthenticated, "username and password must be provided")
}

// StaticAuthUsernameFromContext returns the username authenticated by the static auth plugin and stored in the Context, if any
func StaticAuthUsernameFromContext(ctx context.Context) string {
	username, ok := ctx.Value(staticAuthUsername).(string)
	if ok {
		return username
	}
	return ""
}

func newStaticAuthContext(ctx context.Context, username string) context.Context {
	return context.WithValue(ctx, staticAuthUsername, username)
}

func staticAuthPluginInitializer() (Authenticator, error) {
	entries := make([]StaticAuthConfigEntry, 0)
	if credsFile == "" {
		err := fmt.Errorf("failed to load static auth plugin. Plugin configured but grpc-auth-static-password-file not provided")
		return nil, err
	}

	data, err := os.ReadFile(credsFile)
	if err != nil {
		err := fmt.Errorf("failed to load static auth plugin %v", err)
		return nil, err
	}

	err = json.Unmarshal(data, &entries)
	if err != nil {
		err := fmt.Errorf("fail to load static auth plugin: %v", err)
		return nil, err
	}
	staticAuthPlugin := &StaticAuthPlugin{
		entries: entries,
	}
	log.Info("static auth plugin have initialized successfully with config from grpc-auth-static-password-file")
	return staticAuthPlugin, nil
}

func init() {
	RegisterAuthPlugin("static", staticAuthPluginInitializer)
	grpcAuthServerFlagHooks = append(grpcAuthServerFlagHooks, registerGRPCServerAuthStaticFlags)
}
