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
	"strings"

	"vitess.io/vitess/go/vt/sqlparser"
	"vitess.io/vitess/go/vt/vterrors"
	"vitess.io/vitess/go/vt/vtgate/engine"
	"vitess.io/vitess/go/vt/vtgate/planbuilder/plancontext"
	"vitess.io/vitess/go/vt/vtgate/semantics"
)

// hasWindowFunctions checks if the query contains window functions
func hasWindowFunctions(ctx *plancontext.PlanningContext) bool {
	stmt, ok := ctx.Statement.(*sqlparser.Select)
	if !ok {
		return false
	}

	// Check if there's any OverClause in the entire statement
	var hasWindow bool
	err := sqlparser.Walk(func(node sqlparser.SQLNode) (bool, error) {
		if _, ok := node.(*sqlparser.OverClause); ok {
			hasWindow = true
			return false, nil
		}
		return true, nil
	}, stmt)
	if err != nil {
		return false
	}
	return hasWindow
}

// validateWindowFunctionsForMultiShard validates window functions for multi-shard operations
func validateWindowFunctionsForMultiShard(ctx *plancontext.PlanningContext, op Operator, routes []*Route) error {
	for _, route := range routes {
		if !canExecuteWindowsOnRoute(route) {
			return ctx.SemTable.NotSingleShardErr
		}
		if err := validateWindowPartitions(ctx, route); err != nil {
			return err
		}
		if !canExecuteWindowsInSubqueries(ctx, route) {
			return ctx.SemTable.NotSingleShardErr
		}
	}

	if len(routes) > 0 {
		return nil
	}

	// Check subqueries in the main operator tree if no routes were found
	// This handles cases where window functions are used in UNION ALL or other complex operators
	if canExecuteWindowsInSubqueries(ctx, op) {
		return nil
	}

	return ctx.SemTable.NotSingleShardErr
}

// canExecuteWindowsOnRoute checks if window functions can execute on this route
func canExecuteWindowsOnRoute(route *Route) bool {
	if route == nil {
		return false
	}

	opCode := route.Routing.OpCode()
	// Allow window functions for specific opcodes that can handle them
	if opCode == engine.IN || opCode == engine.EqualUnique || opCode == engine.Unsharded ||
		opCode == engine.None || opCode == engine.DBA {
		return true
	}

	shardedRouting, ok := route.Routing.(*ShardedRouting)
	if !ok {
		// For other non-ShardedRouting types, window functions are not allowed
		return false
	}

	if shardedRouting.Selected == nil {
		return false
	}

	selectedVindex := shardedRouting.Selected.FoundVindex
	if selectedVindex == nil || !selectedVindex.IsUnique() {
		return false
	}

	return true
}

// validateWindowPartitions validates that PARTITION BY includes the sharding key for multi-shard operations
func validateWindowPartitions(ctx *plancontext.PlanningContext, route *Route) error {
	stmt, ok := ctx.Statement.(*sqlparser.Select)
	if !ok {
		return nil
	}

	shardingKey := findShardingKey(route)
	if shardingKey == "" {
		return nil
	}

	for _, selectExpr := range stmt.SelectExprs.Exprs {
		if err := validateWindowPartitionBy(selectExpr, shardingKey); err != nil {
			return err
		}
	}

	if err := validateNamedWindows(stmt.Windows, shardingKey); err != nil {
		return err
	}

	return validateSubqueryWindows(stmt, shardingKey)
}

// canExecuteWindowsInSubqueries checks if subqueries contain valid window functions
func canExecuteWindowsInSubqueries(ctx *plancontext.PlanningContext, op Operator) bool {
	if op == nil || ctx == nil {
		return false
	}

	hasInvalidSubquery := false

	visitF := func(op Operator, _ semantics.TableSet, _ bool) (Operator, *ApplyResult) {
		if op == nil {
			return op, NoRewrite
		}

		if subOp, ok := op.(*SubQuery); ok {
			if subOp == nil || subOp.Subquery == nil {
				hasInvalidSubquery = true
				return op, NoRewrite
			}
			if subRoute, ok := subOp.Subquery.(*Route); ok && subRoute != nil {
				if !canExecuteWindowsOnRoute(subRoute) || validateWindowPartitions(ctx, subRoute) != nil {
					hasInvalidSubquery = true
				}
			} else {
				hasInvalidSubquery = true
			}
		}
		return op, NoRewrite
	}

	TopDown(op, TableID, visitF, stopAtRoute)
	return !hasInvalidSubquery
}

// validateSubqueryWindows validates window functions in subqueries
func validateSubqueryWindows(stmt *sqlparser.Select, shardingKey string) error {
	if stmt.Where == nil {
		return nil
	}

	return sqlparser.Walk(func(node sqlparser.SQLNode) (bool, error) {
		subq, ok := node.(*sqlparser.Subquery)
		if !ok {
			return true, nil
		}
		sel, ok := subq.Select.(*sqlparser.Select)
		if !ok {
			return true, nil
		}
		for _, selectExpr := range sel.SelectExprs.Exprs {
			if err := validateWindowPartitionBy(selectExpr, shardingKey); err != nil {
				return false, err
			}
		}
		return true, validateSubqueryWindows(sel, shardingKey)
	}, stmt.Where)
}

// findShardingKey returns the sharding key from the route
func findShardingKey(route *Route) string {
	shardedRouting, ok := route.Routing.(*ShardedRouting)
	if !ok || len(shardedRouting.VindexPreds) == 0 {
		return ""
	}

	vpp := shardedRouting.VindexPreds[0]
	if vpp.ColVindex != nil && len(vpp.ColVindex.Columns) > 0 {
		return vpp.ColVindex.Columns[0].String()
	}
	return ""
}

// validateNamedWindows validates PARTITION BY clauses in WINDOW definitions
func validateNamedWindows(windows []*sqlparser.NamedWindow, shardingKey string) error {
	for _, namedWindow := range windows {
		for _, windowDef := range namedWindow.Windows {
			if err := validateWindowSpec(windowDef.WindowSpec, shardingKey); err != nil {
				return err
			}
		}
	}
	return nil
}

// validateWindowSpec validates PARTITION BY includes the sharding key
func validateWindowSpec(spec *sqlparser.WindowSpecification, shardingKey string) error {
	if spec == nil || len(spec.PartitionClause) == 0 {
		return vterrors.VT12001("window function PARTITION BY must include the sharding key '" + shardingKey + "' for multi-shard queries")
	}

	for _, expr := range spec.PartitionClause {
		col, ok := expr.(*sqlparser.ColName)
		if !ok {
			return vterrors.VT12001("window function PARTITION BY must include the sharding key '" + shardingKey + "' for multi-shard queries")
		}
		if col.Name.EqualString(shardingKey) || strings.EqualFold(col.Name.String(), shardingKey) {
			return nil
		}
	}

	return vterrors.VT12001("window function PARTITION BY must include the sharding key '" + shardingKey + "' for multi-shard queries")
}

// validateWindowPartitionBy validates window functions in SELECT expressions
func validateWindowPartitionBy(selectExpr interface{}, shardingKey string) error {
	if selectExpr == nil {
		return nil
	}

	var validationErr error
	err := sqlparser.Walk(func(node sqlparser.SQLNode) (bool, error) {
		overClause, ok := node.(*sqlparser.OverClause)
		if !ok {
			return true, nil
		}
		if err := validateWindowSpec(overClause.WindowSpec, shardingKey); err != nil {
			validationErr = err
			return false, nil
		}
		return true, nil
	}, selectExpr.(sqlparser.SQLNode))

	if err != nil {
		return err
	}
	return validationErr
}
