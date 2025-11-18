/*
Copyright 2021 The Vitess Authors.

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

package operators

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"vitess.io/vitess/go/vt/sqlparser"
	"vitess.io/vitess/go/vt/vtgate/planbuilder/plancontext"
)

func TestHasWindowFunctions(t *testing.T) {
	tests := []struct {
		name     string
		sql      string
		expected bool
	}{
		{
			name:     "regular query without window functions",
			sql:      "SELECT id, name FROM users",
			expected: false,
		},
		{
			name:     "query with ROW_NUMBER window function",
			sql:      "SELECT id, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at) FROM orders",
			expected: true,
		},
		{
			name:     "query with named window definition",
			sql:      "SELECT id, ROW_NUMBER() OVER w FROM users WINDOW w AS (PARTITION BY dept_id ORDER BY salary)",
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stmt, err := sqlparser.NewTestParser().Parse(tt.sql)
			assert.NoError(t, err)

			ctx := &plancontext.PlanningContext{
				Statement: stmt,
			}

			result := hasWindowFunctions(ctx)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestValidateWindowSpec(t *testing.T) {
	tests := []struct {
		name        string
		spec        *sqlparser.WindowSpecification
		shardingKey string
		expectError bool
	}{
		{
			name: "window function with sharding key in partition - should pass",
			spec: &sqlparser.WindowSpecification{
				PartitionClause: []sqlparser.Expr{
					&sqlparser.ColName{Name: sqlparser.NewIdentifierCI("user_id")},
				},
			},
			shardingKey: "user_id",
			expectError: false,
		},
		{
			name: "window function partitioned by non-sharding column - should fail",
			spec: &sqlparser.WindowSpecification{
				PartitionClause: []sqlparser.Expr{
					&sqlparser.ColName{Name: sqlparser.NewIdentifierCI("department_id")},
				},
			},
			shardingKey: "user_id",
			expectError: true,
		},
		{
			name: "window function without partition clause - should fail for multi-shard",
			spec: &sqlparser.WindowSpecification{
				PartitionClause: []sqlparser.Expr{},
			},
			shardingKey: "user_id",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateWindowSpec(tt.spec, tt.shardingKey)
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCanExecuteWindowsOnRoute(t *testing.T) {
	tests := []struct {
		name     string
		route    *Route
		expected bool
	}{
		{
			name: "information schema queries allow window functions",
			route: &Route{
				Routing: &InfoSchemaRouting{},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := canExecuteWindowsOnRoute(tt.route)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestValidateWindowPartitionBy(t *testing.T) {
	tests := []struct {
		name        string
		sql         string
		shardingKey string
		expectError bool
	}{
		{
			name:        "window function with correct sharding key in partition",
			sql:         "SELECT ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at) FROM orders",
			shardingKey: "user_id",
			expectError: false,
		},
		{
			name:        "window function partitioned by wrong column fails validation",
			sql:         "SELECT ROW_NUMBER() OVER (PARTITION BY status ORDER BY created_at) FROM orders",
			shardingKey: "user_id",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stmt, err := sqlparser.NewTestParser().Parse(tt.sql)
			assert.NoError(t, err)

			selectStmt := stmt.(*sqlparser.Select)
			err = validateWindowPartitionBy(selectStmt.SelectExprs.Exprs[0], tt.shardingKey)
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
