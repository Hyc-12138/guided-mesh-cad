"""Caller-declared complete source-triangle accounting; not feature recognition."""
import hashlib
import json
from pathlib import Path

ROLES = {'working', 'simplifiable', 'reference', 'unknown', 'excluded'}

def _hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def audit_domains(source, domains, out_dir):
    """Require every original triangle to have one explicit, source-bound owner."""
    from .mesh import load_mesh
    source, domains, out = Path(source).resolve(), Path(domains).resolve(), Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    contract_bytes = domains.read_bytes()
    contract_digest = hashlib.sha256(contract_bytes).hexdigest()
    contract = json.loads(contract_bytes.decode('utf-8-sig'))
    if contract.get('schema') != 'guided-mesh-cad.domains.v1' or contract.get('units') != 'mm':
        raise ValueError('domains require schema guided-mesh-cad.domains.v1 and units mm')
    digest = _hash(source)
    if contract.get('source_sha256') != digest: raise ValueError('source domain hash mismatch')
    _, triangles = load_mesh(source)
    records = contract.get('domains')
    if not isinstance(records, list) or not records: raise ValueError('nonempty domains required')
    owners = [-1] * len(triangles)
    ids = set(); rows = []; errors = []; unknown = excluded = 0
    for i, region in enumerate(records):
        name, role = region.get('id'), region.get('role')
        if not isinstance(name, str) or not name or name in ids: raise ValueError('domain ids must be nonempty and unique')
        ids.add(name)
        if role not in ROLES: raise ValueError('unsupported domain role')
        if not isinstance(region.get('basis'), str) or not region['basis'].strip(): raise ValueError('every domain requires a basis')
        authorization = region.get('authorization')
        if role in ('simplifiable', 'excluded') and (not isinstance(authorization, str) or not authorization.strip()):
            raise ValueError('simplification/exclusion requires explicit caller authorization evidence')
        indices = region.get('triangles')
        if not isinstance(indices, list) or not indices: raise ValueError('each domain requires nonempty original triangle indices')
        local = set()
        for t in indices:
            if isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < len(triangles): raise ValueError('invalid original triangle index')
            if t in local or owners[t] != -1: raise ValueError('ambiguous or duplicate triangle ownership')
            local.add(t); owners[t] = i
        unknown += len(indices) if role == 'unknown' else 0
        excluded += len(indices) if role == 'excluded' else 0
        rows.append({'id':name, 'role':role, 'triangles':len(indices), 'basis':region['basis'], 'authorization':region.get('authorization')})
    missing = [i for i, owner in enumerate(owners) if owner == -1]
    if missing: errors.append('unassigned original source triangles')
    if unknown: errors.append('unknown source domain remains unresolved')
    if _hash(source) != digest or _hash(domains) != contract_digest: raise ValueError('domain audit inputs changed during execution')
    report = {'schema':'guided-mesh-cad.domain-audit.v1', 'status':'coverage_accounted' if not errors else 'coverage_unresolved',
              'source':{'path':str(source), 'sha256':digest}, 'domain_contract':{'path':str(domains), 'sha256':contract_digest},
              'source_triangles':len(triangles), 'unassigned_triangles':len(missing), 'unknown_triangles':unknown,
              'explicitly_excluded_triangles':excluded, 'unassigned_indices':missing, 'regions':rows, 'errors':errors,
              'coverage_complete':not missing, 'domain_accounting_pass':not errors,
              'geometry_acceptance':'not_checked', 'semantics_or_authorization_independently_confirmed':False,
              'scope':'Original source accounting only. Roles and authorization are caller declarations; no geometric or patent inference.'}
    path = out/'domain_audit.json'; path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return {'ok':not errors, 'acceptance':'evidence_only' if not errors else 'unresolved', 'report':report, 'artifacts':{'report':str(path)}}
