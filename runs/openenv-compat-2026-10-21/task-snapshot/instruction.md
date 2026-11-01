Improve a fixed Wi-Fi activity classifier by choosing which target recordings
to label. Request exactly 16 distinct labels from the 24-row query pool.

Use `csi-calibration state` to inspect the current observation and
`csi-calibration query INDEX` to label one available row. Both commands return
JSON. The service URL is in `CSI_CALIBRATION_URL`; its public API is
`GET /state` and `POST /query` with body `{"index": INDEX}`. There is one
episode and no reset. The service owns the classifier and enforces the budget.
You may automate your query strategy.

The observation contains normalized window features, eight classifier scores
per row, requested labels (`-1` for unknown), action and queried masks, and
remaining budget as a fraction. Classes 0–7 are empty room, jumping, standing,
walking, running, squat, clapping, and waving arms. Scores are raw kernel-ridge
outputs. Each label request updates the classifier's scores. Previously
requested rows are unavailable.

The final reward is accuracy after your 16 requests minus source-only accuracy
on 24 separate recording events. It can be negative. An incomplete or invalid
episode receives -1. Reward examples and their labels are private to the
service and verifier. Maximize this accuracy gain through your query choices.
No prediction program or answer file is required: the service records your
accepted requests, and Harbor collects that history for verification.
