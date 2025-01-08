## Extractive Text Summarization and Classification Application

This project is an implementation of **Extractive Text Summarization** using the **TextRank algorithm** and classification of summarized text into categories (`Complaint`, `Fake`, `Help`) using a **Naive Bayes classifier**. The application provides a Flask-based web interface for users to upload files, generate text summaries, classify the content, and view word-frequency visualizations.

---

## Features

1. **Extractive Text Summarization**:
   - Uses the **TextRank algorithm** to extract key sentences from a given text document.
2. **Text Classification**:

   - Classifies summarized text into one of three categories: `Complaint`, `Fake`, or `Help`.
   - Utilizes a pre-trained Naive Bayes classifier for prediction.

3. **Word Frequency Visualization**:

   - Generates bar graphs to display word frequency in different categories for better insights.

4. **Report Generation**:

   - Automatically saves classified summaries into category-specific folders for easy management.

5. **User-Friendly Web Interface**:
   - Allows users to upload files, view extractive summaries, classify text, and generate reports directly from a browser.

---

## Technologies Used

- **Python**: Core programming language.
- **Flask**: Web framework for creating the user interface.
- **Natural Language Toolkit (nltk)**: For text preprocessing, tokenization, stopword removal, and stemming.
- **NetworkX**: For graph-based similarity computation in TextRank.
- **Scikit-learn**: For text vectorization and Naive Bayes classification.
- **Matplotlib**: For generating word-frequency bar graphs.
- **Pickle**: For saving and loading pre-trained models.
- **HTML/CSS**: For designing the web interface.

---

## Installation

Follow the steps below to set up the application:

1. Clone the repository:

   ```bash
   git clone https://github.com/AvaniVaidya/Text-Extraction-And-Classification.git
   cd your-repository
   ```

2. Set up a virtual environment:

   ```
   python -m venv env
   source env/bin/activate  # On Windows: env\\Scripts\\activate
   ```

3. Download necessary NLTK resources: Run the following Python commands to download stopwords and punkt:

   ```
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
   ```

4. Run the application:

   ```
   python app.py
   ```

5. Access the application:
   Open your browser and navigate to http://127.0.0.1:5000.

---

## Usage

- Summarization and Classification:
  - Upload a text file via the web interface.
  - View the generated extractive summary and the predicted category.
- Visualization:

  - View word-frequency bar graphs for different categories.

- Reports:
  - Access saved summaries in the respective category folders.

---

## Contributions

Contributions are welcome! To contribute:

- Fork the repository.
- Create a new branch (git checkout -b feature/your-feature).
- Make your changes and commit them (git commit -am 'Add some feature').
- Push to the branch (git push origin feature/your-feature).
- Create a new Pull Request.

---

## Acknowledgments

- **NLTK**: For providing essential tools for text preprocessing and analysis.
- **Scikit-learn**: For machine learning capabilities.
- **Flask**: For enabling the web interface.
- **NetworkX**: For graph processing used in the TextRank algorithm.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
