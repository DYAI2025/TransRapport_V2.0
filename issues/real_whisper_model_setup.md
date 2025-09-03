# Issue: Real Whisper Model Setup

## Summary
Ensure the application uses real Whisper CT2 weights for speech recognition.

## Tasks
- [ ] Download and configure the `faster-whisper-large-v3` CT2 weights.
- [ ] Test the full ASR pipeline end-to-end with the real model.
- [ ] Validate that the system operates completely offline.

## Acceptance Criteria
- The server loads `faster-whisper-large-v3` from a local path.
- Transcription works for a sample audio file without network access.
- Any attempt to contact external services is disallowed.

## Priority
High
