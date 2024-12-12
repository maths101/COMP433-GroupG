Medical image classification is essential for precise diagnosis, but the lack of labeled datasets is a major obstacle because labeling takes time and skill. Conventional techniques such as CNNs find it difficult to strike a balance between global context and local feature extraction, particularly in pathology where minute variations in tissue patterns are crucial. To overcome these problems, this research suggests utilizing self-supervised learning (SSL) and Vision Transformers (ViTs) in conjunction with the DINO architecture. Using smaller labeled datasets to fine-tune ViTs after pre-training them on unannotated data allows the method to capture both global structures and local details. Preliminary findings indicate that sophisticated medical picture classification can be done with accuracy and precision.

The code is separated into four different files which each do separate tasks for the proposed model. Below are details of each code file and instructions on how to run the code.

## Instructions to Run `Huron_Tiny.ipynb`

### Purpose

This notebook is used to train the encoder that will serve as the backbone for a Self-Supervised Learning (SSL) model.

### Steps to Run

1. **Open in Google Colab**:

   - Upload the notebook file (`Huron_Tiny.ipynb`) to your Google Drive.
   - Open [Google Colab](https://colab.research.google.com/).
   - Click on the **File** menu, then **Open Notebook**, and select the notebook from your Drive.

2. **Grant Drive Access**:

   - When prompted, allow the notebook to access your Google Drive.
   - Ensure the file `huron_unlabeled_data.tar.xz` is present in your Google Drive.

3. **Connect to a Runtime**:

   - In Colab, go to **Runtime** > **Change Runtime Type**.
   - Set the runtime type to `GPU` (for faster training, if available).
   - Then, click **Runtime** > **Connect** to establish a connection to the runtime.

4. **Run All Cells**:

   - Execute all code blocks in the notebook by selecting **Runtime** > **Run All**, or by manually running each block using `Shift + Enter`.

5. **Monitor Progress**:

   - The training process will use the data from `huron_unlabeled_data.tar.xz`.
   - The trained encoder will be saved either to your Google Drive or the directory specified in the notebook.

### Notes

- Ensure you have sufficient storage in your Drive for saving outputs.
- If you encounter issues with file paths, verify that the data file and notebook are in the correct locations within your Drive.

---

## Instructions to Run `Huron_Training.ipynb`

### Purpose

This notebook is used to train the classification head using labeled data.

### Steps to Run

1. **Open in Google Colab**:

   - Upload the notebook file (`Huron_Training.ipynb`) to your Google Drive.
   - Open [Google Colab](https://colab.research.google.com/).
   - Click on the **File** menu, then **Open Notebook**, and select the notebook from your Drive.

2. **Grant Drive Access**:

   - When prompted, allow the notebook to access your Google Drive.
   - Ensure the following datasets are present in your Google Drive:
     - `Huron_Labelled_Data`
     - `Huron_Labelled_Data_4_Class`
     - `Huron_Labelled_Data_2_Class`
   - Upload the `checkpoints` folder containing the file `checkpoint_tiny_epoch_9.pth` to the Google Colab contents folder

3. **Connect to a Runtime**:

   - In Colab, go to **Runtime** > **Change Runtime Type**.
   - Set the runtime type to `GPU` (for faster training, if available).
   - Then, click **Runtime** > **Connect** to establish a connection to the runtime.

4. **Run All Cells**:

   - Execute all code blocks in the notebook by selecting **Runtime** > **Run All**, or by manually running each block using `Shift + Enter`.

5. **Monitor Progress**:

   - The training process will use the labeled datasets and the pre-trained encoder checkpoint.
   - Outputs and updated checkpoints will be saved in your Google Drive or the specified directory in the notebook.

### Notes

- Ensure that the folder structure in your Google Drive matches the paths specified in the notebook.
- Verify the availability of GPU resources for optimal performance.

---

## Instructions to Run `Huron_Validation.ipynb`

### Purpose

This notebook is used to validate models using labeled datasets and pre-trained model checkpoints.

### Steps to Run

1. **Open in Google Colab**:

   - Upload the notebook file (`Huron_Validation.ipynb`) to your Google Drive.
   - Open [Google Colab](https://colab.research.google.com/).
   - Click on the **File** menu, then **Open Notebook**, and select the notebook from your Drive.

2. **Grant Drive Access**:

   - When prompted, allow the notebook to access your Google Drive.
   - Ensure the following datasets are present in your Google Drive:
     - `Huron_Labelled_Data`
     - `Huron_Labelled_Data_4_Class`
     - `Huron_Labelled_Data_2_Class`
   - Upload the `checkpoints` folder containing the following files to the Google Colab contents folder:
     - `final_model_tiny_7_Class.pth`
     - `final_model_tiny_4_Class.pth`
     - `final_model_tiny_2_Class.pth`

3. **Connect to a Runtime**:

   - In Colab, go to **Runtime** > **Change Runtime Type**.
   - Set the runtime type to `GPU` (for faster performance).
   - Then, click **Runtime** > **Connect** to establish a connection to the runtime.

4. **Run All Cells**:

   - Execute all code blocks in the notebook by selecting **Runtime** > **Run All**, or by manually running each block using `Shift + Enter`.

5. **Monitor Progress**:

   - The validation process will use the labeled datasets and pre-trained model checkpoints.
   - Outputs and validation results will be saved either to your Google Drive or the directory specified in the notebook.

### Notes

- Ensure that the folder structure in your Google Drive matches the paths specified in the notebook.
- Verify that the GPU resources are enabled for optimal validation performance.

---

## Instructions to Run `Huron_4_Classes_Validation.ipynb`

### Purpose

This notebook is used to validate models for 4-class classification using the provided sample test labeled dataset with evenly distributed classes.

### Steps to Run

1. **Open in Google Colab**:

   - Upload the notebook file (`Huron_4_Classes_Validation.ipynb`) to your Google Drive.
   - Open [Google Colab](https://colab.research.google.com/).
   - Click on the **File** menu, then **Open Notebook**, and select the notebook from your Drive.

2. **Grant Drive Access**:

   - When prompted, allow the notebook to access your Google Drive.
   - Ensure the following dataset is present in your Google Drive:
     - `Test_Data`
   - Upload the `checkpoints` folder containing the following file to the Google Colab contents folder:
     - `final_model_tiny_4_Class.pth`

3. **Connect to a Runtime**:

   - In Colab, go to **Runtime** > **Change Runtime Type**.
   - Set the runtime type to `GPU` (for faster performance).
   - Then, click **Runtime** > **Connect** to establish a connection to the runtime.

4. **Run All Cells**:

   - Execute all code blocks in the notebook by selecting **Runtime** > **Run All**, or by manually running each block using `Shift + Enter`.

5. **Monitor Progress**:

   - The validation process will use the labeled dataset and the pre-trained model checkpoint.
   - Outputs and validation results will be saved either to your Google Drive or the directory specified in the notebook.

### Notes

- Ensure that the folder structure in your Google Drive matches the paths specified in the notebook.
- Verify that GPU resources are enabled for optimal validation performance.

---

## Instructions to Download the Unlabeled Dataset

### Dataset Location
The unlabeled dataset and the 3 datasets to train the model (Huron_Labelled_Data, Huron_Labelled_Data_4_Class, and Huron_Labelled_Data_2_Class) are hosted on Google Drive. Use the following link to access it:
[Datasets on Google Drive](https://drive.google.com/drive/folders/111_X3SAe6lyAr3rRn5OFupBgAljbUa2I?usp=sharing)

### Steps to Download the Dataset

1. **Open the Link**:
   - Click on the provided link or copy and paste it into your web browser.
   - You will be directed to the Google Drive folder containing the dataset.

2. **Select the Dataset**:
   - Locate the dataset file (e.g., `huron_unlabeled_data.tar.xz`).
   - Right-click on the file to open the context menu.

3. **Download the File**:
   - Click on **Download** from the context menu.
   - If prompted, confirm the download and wait for the file to be saved to your local machine.

4. **Upload to Google Drive**:
   - Log in to your Google Drive account.
   - Click on **+ New** > **File upload**, and select the downloaded dataset file.
   - Ensure the file is uploaded to the correct directory for use with the notebooks.

### Notes
- Ensure you have sufficient storage space available on your local machine and Google Drive.
- If the file is large, a stable internet connection is recommended for downloading and uploading.
- Verify the file's integrity after the download to ensure it is not corrupted.

---

