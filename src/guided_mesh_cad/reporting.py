"""Bounded user-facing summaries. Full evidence stays in on-disk reports."""
import json
import math

MAX_SUMMARY_BYTES = 4096

def short_text(value, limit=320):
    text = ' '.join(str(value).split())
    return text if len(text) <= limit else text[:limit] + '…'

def compact_report(report, *, command=None, report_path=None, log_path=None, ok=None):
    """Do not infer whole geometry acceptance from successful execution."""
    summary = {'schema': 'guided-mesh-cad.summary.v1'}
    if command is not None: summary['command'] = short_text(command, 64)
    summary['status'] = short_text(report.get('status', 'completed' if ok else 'rejected' if ok is False else 'reported'), 80)
    if ok is not None: summary['operation_succeeded'] = bool(ok)
    for key in ('acceptance', 'passed', 'formal_solid_gate_passed', 'package_version',
                'geometry_acceptance', 'scope', 'error_code', 'type', 'error', 'exit_code'):
        if key in report:
            v = report[key]
            if isinstance(v, (bool, int)) or v is None: summary[key] = v
            elif isinstance(v, str): summary[key] = short_text(v)
    counts = {}
    for key in ('resources', 'instances', 'imported_instances', 'source_triangles',
                'unassigned_triangles', 'unknown_triangles', 'completed', 'failed', 'pending', 'reused'):
        v = report.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v): counts[key] = v
    if counts: summary['counts'] = counts
    errors = report.get('errors')
    if isinstance(errors, list):
        summary['errors'] = [short_text(v, 180) for v in errors[:5]]
        summary['additional_errors'] = max(0, len(errors)-5)
    metrics = {}
    for key, value in report.items():
        if len(metrics) >= 8: break
        if key.endswith(('_mm', '_mm2', '_mm3')) and isinstance(value, (int, float)) and math.isfinite(value): metrics[key] = value
    if metrics: summary['metrics'] = metrics
    if report_path: summary['report'] = str(report_path)
    if log_path: summary['log'] = str(log_path)
    # Paths may also be unusually long; never let arbitrary input inflate stdout.
    while len(json.dumps(summary, ensure_ascii=False).encode('utf-8')) > MAX_SUMMARY_BYTES:
        for key in ('metrics', 'errors', 'scope', 'error'):
            if key in summary:
                del summary[key]
                summary['details_omitted'] = True
                break
        else:
            shrunk = False
            for key, value in list(summary.items()):
                if isinstance(value, str) and len(value) > 48:
                    summary[key] = short_text(value, max(48, len(value)//2))
                    if key in ('report', 'log'): summary['path_display_truncated'] = True
                    shrunk = True
            summary['details_omitted'] = True
            if not shrunk: break
    return summary

def print_report(report, *, full=False, **kwargs):
    value = report if full else compact_report(report, **kwargs)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
