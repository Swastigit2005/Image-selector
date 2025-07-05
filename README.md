# Image-selector
This project is an automated wedding image selector that intelligently filters, scores, and selects the best photos from a large set of wedding images. It uses advanced computer vision and deep learning models to evaluate each image based on multiple criteria, including:
•	Face and person detection: Ensures real people are present and filters out images with too few or low-quality faces.
•	Face quality assessment: Checks sharpness, pose, eye openness, and other facial attributes.
•	Aesthetic and composition scoring: Uses models like CLIP and custom logic to rate visual appeal, color harmony, rule of thirds, symmetry, and more.
•	Background clutter detection: Penalizes images with messy or distracting backgrounds.
•	Emotion recognition: Prefers images with positive emotions.
•	Exposure and sharpness checks: Filters out blurry or poorly exposed images.
•	Deduplication: Groups similar images and selects the best from each group to avoid near-duplicates.
•	Final selection and export: Outputs the top N images and logs results to CSV.
The main pipeline is orchestrated in main.py, which ties together all scoring, filtering, deduplication, and export steps. This tool is ideal for photographers or event organizers who want to quickly find the best wedding photos from thousands of raw images.

