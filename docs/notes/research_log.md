# AFSACPSO Research Log

## Current Retained Stack

The repository now keeps only the reduced AFSACPSO core built around:
- `AttractorFieldControl`
- `CrossScaleState`
- `CrossQ-SAC`

The removed exploratory branches, reward add-ons, and uncertainty-penalty experiments were intentionally pruned from the active code paths so the shipped implementation reflects only the retained stack.
