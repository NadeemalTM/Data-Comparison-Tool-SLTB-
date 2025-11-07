# CSVDash Thorough Testing Plan

## Completed Fixes
- [x] Resolve langchain-core version conflicts
- [x] Install missing matplotlib package
- [x] Remove deprecated Streamlit config option
- [x] Add error handling for missing CSV file on home page
- [x] Temporarily disable LangChain integration
- [x] Fix StreamlitDuplicateElementId error by adding unique keys to plotly_chart elements
- [x] Fix comparison errors: remove premature string conversion, simplify column intersection, improve metric labels, convert join columns to string to prevent merge/datacompy errors, convert all columns to string for consistent comparison, add safety checks for join column existence

## Testing Tasks
- [x] Launch browser and navigate to http://localhost:8501
  - [x] Verified app is running and serving HTML content
- [ ] Test Home page:
  - [ ] Verify page title and introduction
  - [ ] Check Anime_CSV link
  - [ ] Verify UI Changes section
  - [ ] Verify UAT Progress section
  - [ ] Verify What you can do section
  - [ ] Check CSV data display (should show sample data or graceful error)
- [ ] Test navigation to File page
- [ ] Test File page:
  - [ ] Verify page loads without errors
  - [ ] Test file upload functionality (upload a valid CSV)
  - [ ] Test data processing after upload
  - [ ] Test querying functionality (UI should be present, even if backend is disabled)
  - [ ] Test data visualization features
- [ ] Test edge cases:
  - [ ] Upload invalid file types (non-CSV)
  - [ ] Upload empty CSV file
  - [ ] Test with large CSV file (if available)
  - [ ] Test navigation back to Home page
- [ ] Verify no console errors in browser
- [ ] Test responsiveness (if applicable)

## Notes
- LangChain integration is temporarily disabled due to version conflicts
- Home page handles missing sample CSV gracefully
- App should run smoothly without crashes
