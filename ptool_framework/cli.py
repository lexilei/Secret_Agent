"""
CLI: Command-line interface for ptool framework.

Commands:
    ptool generate <task> --output <file>   Generate program from task description
    ptool run <program> --input <data>      Run program with trace collection
    ptool refactor <file> --mode <mode>     Refactor program (distill/expand)
    ptool analyze <file>                    Analyze distillation opportunities
    ptool dashboard [--port <port>]         Launch observability dashboard
    ptool traces [--ptool <name>]           View collected traces
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional
import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)

# Try to import click/typer
try:
    import click
    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False


def _ensure_click():
    if not HAS_CLICK:
        print("Error: 'click' package required. Install with: pip install click")
        sys.exit(1)


# ============================================================================
# CLI Application
# ============================================================================

if HAS_CLICK:

    @click.group()
    @click.version_option(version="0.2.0")
    def cli():
        """ptool: Python-Calling-LLMs Agent Framework

        Generate, run, and optimize programs where Python controls flow
        and LLMs handle reasoning.
        """
        pass

    # ========================================================================
    # Generate Command
    # ========================================================================

    @cli.command()
    @click.argument("task_description")
    @click.option("--output", "-o", help="Output file path")
    @click.option("--template", "-t", type=click.Choice(["analyzer", "classifier", "extractor"]),
                  help="Use a predefined template")
    @click.option("--model", "-m", default="deepseek-v3-0324", help="Default LLM model")
    def generate(task_description: str, output: Optional[str], template: Optional[str], model: str):
        """Generate a Python program from a task description.

        Examples:
            ptool generate "Analyze customer reviews" -o review_analyzer.py
            ptool generate "Classify emails" --template classifier -o classifier.py
        """
        from .program_generator import ProgramGenerator, generate_from_template

        if template:
            # Use template
            code = generate_from_template(template, task_description, output)
            if not output:
                print(code)
            else:
                print(f"Generated {output} from template '{template}'")
        else:
            # Full generation via LLM
            generator = ProgramGenerator(default_model=model)
            result = generator.generate(task_description, output)

            if not output:
                print(result.full_code)
            else:
                print(f"Generated {output}")
                print(f"  - {len(result.ptools)} ptools defined")

    # ========================================================================
    # Run Command
    # ========================================================================

    @cli.command()
    @click.argument("program_path", type=click.Path(exists=True))
    @click.option("--input", "-i", "input_data", help="Input data or file")
    @click.option("--file", "-f", "input_file", type=click.Path(exists=True), help="Read input from file")
    @click.option("--collect-traces/--no-traces", default=True, help="Collect execution traces")
    def run(program_path: str, input_data: Optional[str], input_file: Optional[str], collect_traces: bool):
        """Run a ptool program with optional trace collection.

        Examples:
            ptool run analyzer.py --input "Great product!"
            ptool run analyzer.py --file reviews.txt
        """
        from .llm_backend import enable_tracing

        enable_tracing(collect_traces)

        # Get input
        if input_file:
            with open(input_file) as f:
                data = f.read()
        elif input_data:
            data = input_data
        else:
            click.echo("Reading from stdin...", err=True)
            data = sys.stdin.read()

        # Run the program
        import subprocess
        result = subprocess.run(
            [sys.executable, program_path, data],
            capture_output=True,
            text=True,
        )

        if result.stdout:
            click.echo(result.stdout)
        if result.stderr:
            click.echo(result.stderr, err=True)

        if collect_traces:
            from .trace_store import get_trace_store
            store = get_trace_store()
            stats = store.get_stats()
            click.echo(f"\nTraces collected: {stats['session_traces']}", err=True)

        sys.exit(result.returncode)

    # ========================================================================
    # Refactor Command
    # ========================================================================

    @cli.command()
    @click.argument("source_path", type=click.Path(exists=True))
    @click.option("--mode", "-m", type=click.Choice(["distill", "expand"]), required=True,
                  help="distill: LLM→Python, expand: Python→LLM")
    @click.option("--output", "-o", help="Output file path")
    @click.option("--min-traces", default=10, help="Minimum traces for distillation")
    @click.option("--functions", "-f", multiple=True, help="Specific functions to refactor")
    @click.option("--dry-run", is_flag=True, help="Show what would be done without writing")
    @click.option("--analyze-only", is_flag=True, help="Only analyze, don't refactor")
    def refactor(source_path: str, mode: str, output: Optional[str], min_traces: int,
                 functions: tuple, dry_run: bool, analyze_only: bool):
        """Refactor a program to be more or less agentic.

        distill: Convert @ptool LLM calls to @distilled pure Python (where traces support)
        expand: Convert pure Python functions to @ptool for more flexibility

        Examples:
            ptool refactor analyzer.py --mode distill -o analyzer_v2.py
            ptool refactor parser.py --mode expand -o flexible_parser.py
            ptool refactor program.py --mode distill --analyze-only
            ptool refactor program.py --mode distill -f extract_items -f classify
        """
        from .refactorer import CodeRefactorer, analyze_program

        output_path = output or source_path.replace(".py", f"_{mode}ed.py")
        target_functions = list(functions) if functions else None

        # Create refactorer
        refactorer = CodeRefactorer(min_traces_for_distill=min_traces)

        # Analyze first
        click.echo(f"\nAnalyzing {source_path}...")
        analysis = refactorer.analyze(source_path)

        click.echo(f"\n=== Source Analysis ===")
        click.echo(f"Total functions: {analysis['total_functions']}")
        click.echo(f"@ptool functions: {len(analysis['ptool_functions'])}")
        click.echo(f"@distilled functions: {len(analysis['distilled_functions'])}")
        click.echo(f"Pure Python functions: {len(analysis['pure_functions'])}")

        if mode == "distill":
            candidates = analysis['distill_candidates']
            if not candidates:
                click.echo("\nNo ptools have enough traces for distillation.")
                click.echo(f"Need at least {min_traces} traces per ptool.")
                return

            click.echo(f"\n=== Distillation Candidates ({len(candidates)}) ===")
            for c in candidates:
                status = c['recommendation'].upper()
                click.echo(f"  {c['name']}: {c['trace_count']} traces, "
                          f"{c['estimated_coverage']:.0%} coverage → {status}")
                if c.get('reason'):
                    click.echo(f"    {c['reason']}")

        else:  # expand
            candidates = analysis['expand_candidates']
            if not candidates:
                click.echo("\nNo functions identified for expansion.")
                return

            click.echo(f"\n=== Expansion Candidates ({len(candidates)}) ===")
            for c in candidates:
                click.echo(f"  {c['name']}: confidence {c['confidence']:.0%}")
                click.echo(f"    Reason: {c['reason']}")

        if analyze_only:
            click.echo("\nAnalysis complete (--analyze-only specified).")
            return

        if dry_run:
            click.echo("\nDry run - showing what would be refactored...")

        # Perform refactoring
        click.echo(f"\n=== Refactoring ({mode}) ===")
        result = refactorer.refactor(
            source_path=source_path,
            output_path=output_path,
            mode=mode,
            functions=target_functions,
            dry_run=dry_run,
        )

        # Report results
        successful = [c for c in result.changes if c.success]
        failed = [c for c in result.changes if not c.success]

        if successful:
            click.echo(f"\nSuccessfully refactored {len(successful)} function(s):")
            for c in successful:
                click.echo(f"  ✓ {c.function_name}")

        if failed:
            click.echo(f"\nFailed to refactor {len(failed)} function(s):")
            for c in failed:
                click.echo(f"  ✗ {c.function_name}: {c.error}")

        if result.compile_success:
            click.echo(f"\n✓ Output compiles successfully")
        else:
            click.echo(f"\n✗ Compile error: {result.error}")

        if not dry_run and result.compile_success:
            click.echo(f"\nWrote refactored code to: {output_path}")

    # ========================================================================
    # Analyze Command
    # ========================================================================

    @cli.command()
    @click.argument("program_path", type=click.Path(exists=True), required=False)
    @click.option("--min-traces", default=10, help="Minimum traces for analysis")
    def analyze(program_path: Optional[str], min_traces: int):
        """Analyze distillation opportunities for ptools.

        Shows which ptools have enough traces for distillation and their
        estimated pattern coverage.

        Examples:
            ptool analyze
            ptool analyze analyzer.py --min-traces 50
        """
        from .trace_store import get_trace_store
        from .distilled import analyze_fallbacks, get_fallback_log

        store = get_trace_store()
        stats = store.get_stats()

        click.echo("\n=== Trace Statistics ===")
        click.echo(f"Total traces: {stats['total_traces']}")
        click.echo(f"PTools tracked: {stats['ptool_count']}")
        click.echo(f"Storage path: {stats['storage_path']}")

        click.echo("\n=== PTools with Traces ===")
        for name, count in sorted(stats.get('ptools', {}).items(), key=lambda x: -x[1]):
            traces = store.get_traces(ptool_name=name, limit=1000)
            success_count = sum(1 for t in traces if t.success)
            rate = success_count / len(traces) if traces else 0

            status = ""
            if count >= min_traces and rate >= 0.9:
                status = "✓ DISTILLABLE"
            elif count >= min_traces and rate >= 0.7:
                status = "◐ PARTIAL"
            elif count >= min_traces:
                status = "⚠ LOW SUCCESS"
            else:
                status = f"✗ Need {min_traces - count} more"

            click.echo(f"  {name}: {count} traces, {rate:.0%} success {status}")

        # Fallback analysis
        fallbacks = get_fallback_log()
        if fallbacks:
            click.echo(f"\n=== Recent Fallbacks ({len(fallbacks)}) ===")
            analysis = analyze_fallbacks()
            for ptool_name, data in analysis.items():
                click.echo(f"  {ptool_name}: {data['total_fallbacks']} fallbacks")
                for reason, count in data['top_reasons'][:3]:
                    click.echo(f"    - {reason}: {count}")

    # ========================================================================
    # Dashboard Command
    # ========================================================================

    @cli.command()
    @click.option("--host", "-h", default="127.0.0.1", help="Host to bind to")
    @click.option("--port", "-p", default=8080, help="Port to listen on")
    @click.option("--reload", is_flag=True, help="Enable auto-reload (development)")
    def dashboard(host: str, port: int, reload: bool):
        """Launch the observability dashboard.

        Opens a web interface for monitoring ptool execution, viewing traces,
        and analyzing LLM interactions.

        Example:
            ptool dashboard --port 8080
        """
        from .dashboard.server import run_server
        run_server(host=host, port=port, reload=reload)

    # ========================================================================
    # Traces Command
    # ========================================================================

    @cli.command()
    @click.option("--ptool", "-p", help="Filter by ptool name")
    @click.option("--limit", "-n", default=20, help="Number of traces to show")
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON")
    @click.option("--success-only", is_flag=True, help="Only show successful traces")
    def traces(ptool: Optional[str], limit: int, as_json: bool, success_only: bool):
        """View collected execution traces.

        Examples:
            ptool traces --limit 10
            ptool traces --ptool extract_food_items --json
        """
        from .trace_store import get_trace_store

        store = get_trace_store()
        trace_list = store.get_traces(
            ptool_name=ptool,
            limit=limit,
            success_only=success_only,
        )

        if as_json:
            click.echo(json.dumps([t.to_dict() for t in trace_list], indent=2, default=str))
            return

        if not trace_list:
            click.echo("No traces found.")
            return

        click.echo(f"\n=== Traces ({len(trace_list)}) ===\n")
        for t in trace_list:
            status = "✓" if t.success else "✗"
            click.echo(f"{status} {t.ptool_name} [{t.trace_id}]")
            click.echo(f"  Time: {t.timestamp}")
            click.echo(f"  Duration: {t.execution_time_ms:.1f}ms")
            click.echo(f"  Model: {t.model_used}")
            if t.error:
                click.echo(f"  Error: {t.error}")
            else:
                output_str = str(t.output)
                if len(output_str) > 100:
                    output_str = output_str[:100] + "..."
                click.echo(f"  Output: {output_str}")
            click.echo()

    # ========================================================================
    # Models Command
    # ========================================================================

    @cli.command()
    def models():
        """List available LLM models from LLMS.json."""
        from .llm_backend import list_available_models, get_model_info

        models = list_available_models()
        click.echo(f"\n=== Available Models ({len(models)}) ===\n")

        for name in models:
            info = get_model_info(name)
            cost = info['cost']
            if cost == 'local':
                cost_str = "local (free)"
            elif isinstance(cost, dict):
                cost_str = f"${cost.get('input', '?')}/{cost.get('output', '?')} per 1K tokens"
            else:
                cost_str = str(cost)

            click.echo(f"  {name}")
            click.echo(f"    Provider: {info['provider']}")
            click.echo(f"    Cost: {cost_str}")
            if info.get('capabilities'):
                click.echo(f"    Capabilities: {', '.join(info['capabilities'][:5])}")
            click.echo()


# ============================================================================
# Main entry point
# ============================================================================

def main():
    """Main entry point for CLI."""
    _ensure_click()
    cli()


if __name__ == "__main__":
    main()
