# Data provenance and schema

`fertilizer_sample.csv` contains **96 synthetic demonstration records**, generated with Python random seed 42. There are 24 records per artificial class: Urea, DAP, MOP and NPK 19-19-19. Class-specific nutrient centers plus random jitter create easy software demonstration patterns. They are not agronomically validated rules. Crop/soil categories cycle independently of the label.

These records are NOT observations, field trials, research evidence or a fertilizer prescription. Scores on them only test the software pipeline.

Required columns: `Nitrogen,Phosphorus,Potassium,Temperature,Humidity,Moisture,pH,Crop,Soil_Type,Fertilizer`.

Demo conventions: N/P/K in mg/kg, temperature in °C, humidity and moisture in %, pH dimensionless. Moisture measurement method and nutrient extraction method must be documented and consistent in a real dataset; these synthetic values imply no measurement protocol. Allowed software ranges are N/P/K 0–500, temperature −10–60, humidity/moisture 0–100, pH 0–14. These broad input bounds are not agronomic suitability limits.

Training requires finite measurements, nonempty categorical text, at least two classes and at least two records per class, with enough examples in both stratified partitions. Duplicate feature rows are rejected. Replace with appropriately sourced, licensed and expert-labeled observations before research use; document region, crop stage, methods and units. Use field/site/season-separated evaluation to avoid leakage. Mark any other synthetic file with `--synthetic`.

Pest images and YOLO labels are not included. See the root README for dataset structure.
