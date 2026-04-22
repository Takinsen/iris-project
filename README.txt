   This code provides a baseline iris verification evaluation framework using the Open-Iris / Worldcoin pipeline. It selects valid images, generates templates, performs automatic pairwise matching, and outputs distance matrices, summary metrics, and confusion matrices.

----------------------------------------------------------------------------------------
1. Requirement 
	- Anaconda prompt
	- Python 3.11
	- iris-dev ( open-iris environment in Anaconda )
	- iris dataset ( clean black & white up to usage of worldcoin )
	- numpy, pandas, matplotlib
----------------------------------------------------------------------------------------
2. Limitations
	- Currently designed for CASIA-Iris-Thousand ( Dataset follows subject/eye-side folder structure )
	- Currently designed to uses one eye side per run
----------------------------------------------------------------------------------------
3. How to run 
	1. Open anaconda prompt
	2. Activate environment	-> type "conda activate iris_dev" 
	3. Go to the project directory -> type "cd <location of project folder>
	4. Run the script -> type "python <file name we gonna run>.py"
----------------------------------------------------------------------------------------
4. Configuration
	1. DATASET_ROOT -> Path to the CASIA-Iris-Thousand dataset
	2. OUTPUT_ROOT -> Folder where experiment results will be saved
	3. CACHE_DIR -> Folder used to store cached iris templates
	4. TARGET_EYE_SIDE -> Eye side used in the experiment (L or R)
	5. SUBJECTS_PER_SET -> Number of subjects in each experiment set
	6. IMAGES_PER_SUBJECT -> Number of valid images required for each subject
	7. MAX_SETS -> Maximum number of sets to run (None = run all possible sets)
	8. THRESHOLD -> Hamming distance threshold used for match / non-match decision
	9. USE_CACHE -> If True, reuse existing cached templates
	10. OVERWRITE_CACHE -> If True, regenerate cached templates even if they already exist
----------------------------------------------------------------------------------------
5. Output
	- selected_images.csv -> images successfully used in the experiment
	- failed_images.csv -> images that failed template creation
	- skipped_subjects.csv -> subjects excluded because not enough valid images were available
	- set_manifest.csv -> records the subjects/images included in each set
	- distance_matrix.csv -> pairwise Hamming distance matrix
	- pair_records.csv -> unique pairwise comparisons and labels
	- comparison_summary.csv -> summary metrics for the set
	- confusion_matrix.csv -> confusion matrix table
	- confusion_matrix.png -> confusion matrix visualization
	- all_sets_summary.csv -> summary across all sets
	- aggregate_summary.csv -> average/statistics across sets
----------------------------------------------------------------------------------------
6. Evaluation Metrics 
	1. TP (True Positive) -> Genuine pairs correctly predicted as match
	2. FP (False Positive) -> Impostor pairs incorrectly predicted as match
	3. FN (False Negative) -> Genuine pairs incorrectly predicted as non-match
	4. TN (True Negative) -> Impostor pairs correctly predicted as non-match
	5. Accuracy -> Overall proportion of correct predictions
	6. Precision -> Proportion of predicted matches that are actually genuine
	7. Recall -> Proportion of genuine pairs that are correctly detected
	8. F1-score -> Harmonic mean of precision and recall
	9. Balanced Accuracy -> Average of recall and true negative rate; useful when the number of genuine and impostor pairs is imbalanced
----------------------------------------------------------------------------------------
Note 
   Each set uses a fixed number of subjects and a fixed number of images per subject. If an image fails template creation, it is skipped. If a subject cannot provide the required number of valid images, that subject is excluded from the set.
----------------------------------------------------------------------------------------
   At the end, i hope this help and provide efficient convenient to you guys. Good luck & Have fun.