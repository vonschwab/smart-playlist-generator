# TODO & Roadmap

**Last Updated:** 2026-01-10

---

## Current Status

All major features and diagnostics from Phase 2 (DJ Bridging with Genre Integration) are complete and tested.

---

## 🟢 Completed (Phase 2)

### DJ Pool Diagnostics
- ✅ Always-on pool visibility logging
- ✅ Per-track membership tracking (L+T+G format)
- ✅ Waypoint saturation metrics
- ✅ Genre pool made opt-in (k_genre=0 default)

### Code Quality
- ✅ IDF consistency in waypoint scoring
- ✅ Raw genre vector normalization for coverage
- ✅ Progress_arc override disable support
- ✅ Post-order recency validation removed
- ✅ GUI logging to project directory
- ✅ All diagnostics archived

---

## 🔵 Pending

### Testing & Validation
- **Regression Tests**: Add unit tests for DJ pool diagnostics
- **Integration Tests**: Validate end-to-end DJ bridging scenarios

### Documentation
- **Update USER_GUIDE.md**: Document DJ bridging features and configuration
- **Update CONFIG.md**: Add Phase 2 configuration parameters
- **Update CHANGELOG**: Finalize Phase 2 release notes

### Pool Optimization (Future)
- **S1_local**: Evaluate if k_local=200 is optimal
- **S2_toward**: Evaluate if k_toward=80 provides sufficient diversity
- **Saturation Analysis**: Use metrics to tune waypoint_cap and weights

---

## 📋 Future Enhancements

### DJ Bridging
- Genre path visualization in audit reports
- User-specified genre waypoint hints
- A/B comparison mode (baseline vs dj_union side-by-side)

### Configuration
- Config validation on startup
- Migration tool for deprecated keys
- Preset configurations for common use cases

### Performance
- Caching optimization for large libraries
- Parallel segment processing

---

## 📊 Metrics & Success Criteria

### Phase 2 Goals (✅ Met)
- Genre pool contribution visible and debuggable
- Waypoint saturation measurable (<30% near-cap)
- Pool overlaps quantified (L∩T, L∩G, T∩G)
- All diagnostics working without breaking existing tests

### Next Phase Goals
- Test coverage >80% for DJ bridging modules
- Documentation complete and up-to-date
- Pool parameters optimized based on diagnostic data

---

## 🔗 Related Documentation

- `ARCHITECTURE.md` - System architecture overview
- `DJ_BRIDGE_ARCHITECTURE.md` - Complete DJ bridging design
- `CHANGELOG_Phase2.md` - Phase 2 implementation details
- `CONFIG.md` - Configuration reference
- `docs/diagnostics/` - Active diagnostic reports (currently empty)
- `docs/archive/diagnostics_2026-01/` - Archived diagnostics and audits
