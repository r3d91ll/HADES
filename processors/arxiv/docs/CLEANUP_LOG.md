# Cleanup Log - January 2025

## Scripts Directory Cleanup

### Scripts Kept (2 files)
1. **rebuild_dual_gpu.py** - Currently running the database rebuild
2. **process_pdf_on_demand.py** - Essential for on-demand PDF processing

### Scripts Moved to Acheron (7 files)
All moved with timestamp: 2025-01-20_13-56-48

#### Superseded Rebuild Scripts
1. **rebuild_with_jina_v4.py** - Legacy, superseded by rebuild_from_source
2. **rebuild_from_source.py** - Single GPU version, superseded by dual-GPU
3. **test_dual_gpu_simple.py** - Multiprocessing fallback, not needed
4. **verify_dual_gpu.py** - One-time verification script

#### Outdated Utility Scripts  
5. **check_and_rebuild.py** - Not compatible with v4 structure
6. **daily_arxiv_update.py** - Needs rewrite for Jina v4
7. **catchup_arxiv.py** - Needs rewrite for Jina v4

## Rationale

### Why These Were Removed
- **Rebuild scripts**: We have a working dual-GPU solution with Ray
- **Test scripts**: Initial setup complete, no longer needed
- **Update scripts**: Written for v3, need complete rewrite for v4 API

### Future Work Needed
- Rewrite daily update scripts for Jina v4
- Create new catchup mechanism compatible with v4
- Implement incremental processing pipeline

## Directory Structure After Cleanup

```
scripts/
├── rebuild_dual_gpu.py       # Active rebuild (running)
└── process_pdf_on_demand.py  # On-demand processing
```

From 9 scripts down to 2 essential ones.

## Acheron Archive

All deprecated scripts preserved in:
```
/home/todd/olympus/Acheron/HADES/processors/arxiv/scripts/
```

With timestamp suffix: `_2025-01-20_13-56-48.py`

## Impact

- Cleaner, more maintainable codebase
- Clear separation of active vs deprecated code
- Easier to understand what's actually being used
- Historical code preserved for reference

---

*Cleanup completed: January 20, 2025*
*Active rebuild: In progress with rebuild_dual_gpu.py*