"""
Domain-specific audits for ptool_framework.

This package contains domain-specific audit implementations that inherit
from the generic audit base classes.

Available Domains:
    - medcalc: Medical calculator audits (BMI, GFR, etc.)

Example:
    >>> from ptool_framework.audit.domains.medcalc import MedCalcStructureAudit
    >>>
    >>> audit = MedCalcStructureAudit()
    >>> report = audit.run(df, metadata)
"""

# Import domain modules as they're created
# from .medcalc import (...)
