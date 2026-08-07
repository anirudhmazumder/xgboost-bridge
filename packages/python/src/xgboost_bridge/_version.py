"""Single source of the package version.

Separate module rather than an assignment in ``__init__``: ``export`` stamps
``provenance.exporter_version`` at module scope, so if the version lived in the
package ``__init__`` the import order between the two would decide whether
importing this package works at all. A leaf module both can import has no such
ordering hazard, and nothing about it depends on statement order surviving a
future edit.
"""

from __future__ import annotations

__version__ = "1.0.0rc2"
