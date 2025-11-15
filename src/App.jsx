import React, { useState } from 'react';
import axios from 'axios';

const OutcomeCalculator = () => {
  const API_URL = import.meta.env.VITE_API_BASE_URL;
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    primaryPayer: '',
    householdIncome: '',
    icdMethod: 'manual', // user has choice between 'manual' or 'upload'
    icdCodes: ['', '', '', '', ''],
    uploadedFile: null
  });

  const [results, setResults] = useState({
    mortality30: '',
    readmission30: ''
  });

  const [icdSearchResults, setIcdSearchResults] = useState([]);

  const primaryPayerOptions = [
    'Medicare',
    'Medicaid',
    'Private Insurance',
    'Self-Pay',
    'Workers Compensation',
    'Other'
  ];

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleIcdCodeChange = async (index, value) => {
    const newCodes = [...formData.icdCodes];
    newCodes[index] = value;
    setFormData(prev => ({
      ...prev,
      icdCodes: newCodes
    }));

    if (value.length > 2) {
      try {
        const response = await axios.get(`${API_URL}/search_icd/?q=${value}`);
        setIcdSearchResults(response.data);
      } catch (error) {
        console.error("Error searching for ICD codes:", error);
      }
    } else {
      setIcdSearchResults([]);
    }
  };

  const selectIcdCode = (index, code) => {
    const newCodes = [...formData.icdCodes];
    newCodes[index] = code;
    setFormData(prev => ({
      ...prev,
      icdCodes: newCodes
    }));
    setIcdSearchResults([]);
  };

  const addMoreCodes = () => {
    setFormData(prev => ({
      ...prev,
      icdCodes: [...prev.icdCodes, '']
    }));
  };

  const deleteIcdCode = (index) => {
    // does not allow user to have 0 codes
    if (formData.icdCodes.length <= 1) return;
    
    const newCodes = formData.icdCodes.filter((_, i) => i !== index);
    setFormData(prev => ({
      ...prev,
      icdCodes: newCodes
    }));
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) {
      return;
    }

    setFormData(prev => ({
      ...prev,
      uploadedFile: file
    }));

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/upload_icd_file/`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setFormData(prev => ({
        ...prev,
        icdCodes: response.data
      }));
    } catch (error) {
      console.error("Error uploading file:", error);
      alert("There was an error processing the uploaded file.");
    }
  };

  const calculateRisk = async () => {
    const { age, gender, primaryPayer, householdIncome, icdCodes } = formData;

    const pay1Mapping = {
      'Medicare': 1,
      'Medicaid': 2,
      'Private Insurance': 3,
      'Self-Pay': 4,
      'Workers Compensation': 5,
      'Other': 6
    };

    const payload = {
      age: parseInt(age),
      female: gender === 'F' ? 1 : 0,
      pay1: pay1Mapping[primaryPayer],
      zipinc_qrtl: parseInt(householdIncome),
      icd_codes: icdCodes.filter(code => code.trim() !== '')
    };

    try {
      const response = await axios.post(`${API_URL}/predict/`, payload);
      const { prediction, interpretation } = response.data;
      setResults({
        readmission30: `${(prediction * 100).toFixed(1)}%`,
        mortality30: 'N/A' // The model does not predict mortality
      });
    } catch (error) {
      console.error("Error calculating risk:", error);
      alert("There was an error calculating the risk. Please check the inputs and try again.");
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <div className="header">
        <div className="header-content">
          <div className="title">
            <div className="subtitle">Brown University Medical School</div>
            <h1 className="main-title">ICD Diagnosis Code Prediction Calculator</h1>
          </div>
        </div>
      </div>

      <div className="main-content">
        {/* Input Side */}
        <div className="input-panel">
          <h2 className="panel-title">Enter the following data:</h2>
          
          <div className="form-container">
            {/* Age */}
            <div className="form-row">
              <label className="form-label">Age:</label>
              <input
                type="number"
                placeholder="years"
                className="form-input"
                value={formData.age}
                onChange={(e) => handleInputChange('age', e.target.value)}
              />
            </div>

            {/* Gender */}
            <div className="form-row">
              <label className="form-label">Gender:</label>
              <div className="button-group">
                <button
                  className={`toggle-button ${formData.gender === 'M' ? 'active' : ''}`}
                  onClick={() => handleInputChange('gender', 'M')}
                >
                  M
                </button>
                <button
                  className={`toggle-button ${formData.gender === 'F' ? 'active' : ''}`}
                  onClick={() => handleInputChange('gender', 'F')}
                >
                  F
                </button>
              </div>
            </div>

            {/* Expected Primary Payer */}
            <div className="form-row">
              <label className="form-label">Expected Primary Payer:</label>
              <select
                className="form-select"
                value={formData.primaryPayer}
                onChange={(e) => handleInputChange('primaryPayer', e.target.value)}
              >
                <option value="">Select...</option>
                {primaryPayerOptions.map(option => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </div>

            {/* Household Income Quartile */}
            <div className="form-row">
              <label className="form-label">Household Income Quartile:</label>
              <div className="button-group">
                {[1, 2, 3, 4].map(quartile => (
                  <button
                    key={quartile}
                    className={`toggle-button ${formData.householdIncome === quartile.toString() ? 'active' : ''}`}
                    onClick={() => handleInputChange('householdIncome', quartile.toString())}
                  >
                    {quartile}
                  </button>
                ))}
              </div>
            </div>

            {/* ICD-10-CM Codes */}
            <div className="form-row icd-row">
              <label className="form-label">ICD-10-CM Codes:</label>
              <div className="icd-container">
                <div className="radio-group">
                  <label className="radio-label">
                    <input
                      type="radio"
                      name="icdMethod"
                      checked={formData.icdMethod === 'manual'}
                      onChange={() => handleInputChange('icdMethod', 'manual')}
                    />
                    Manual Input
                  </label>
                  <label className="radio-label">
                    <input
                      type="radio"
                      name="icdMethod"
                      checked={formData.icdMethod === 'upload'}
                      onChange={() => handleInputChange('icdMethod', 'upload')}
                    />
                    Upload File
                  </label>
                </div>

                {formData.icdMethod === 'manual' ? (
                  <div className="manual-input-container">
                    <div className="icd-codes">
                      {formData.icdCodes.map((code, index) => (
                        <div key={index} className="icd-code-row">
                          <input
                            type="text"
                            placeholder={`Code ${index + 1}`}
                            className="icd-input"
                            value={code}
                            onChange={(e) => handleIcdCodeChange(index, e.target.value)}
                          />
                          {icdSearchResults && Object.keys(icdSearchResults).length > 0 && (
                            <div className="search-results">
                              {Object.entries(icdSearchResults).map(([code, desc]) => (
                                <div
                                  key={code}
                                  className="search-result"
                                  onClick={() => selectIcdCode(index, code)}
                                >
                                  {code}: {desc}
                                </div>
                              ))}
                            </div>
                          )}
                          {formData.icdCodes.length > 1 && (
                            <button
                              onClick={() => deleteIcdCode(index)}
                              className="delete-icd-button"
                              title="Delete this ICD code"
                            >
                              ×
                            </button>
                          )}
                        </div>
                      ))}
                    </div>
                    <button
                      onClick={addMoreCodes}
                      className="add-more-button"
                    >
                      Add More
                    </button>
                  </div>
                ) : (
                  <div className="upload-container">
                    <input
                      type="file"
                      accept=".csv,.txt,.xlsx"
                      onChange={handleFileUpload}
                      className="file-input"
                    />
                    {formData.uploadedFile && (
                      <p className="file-name">
                        Selected: {formData.uploadedFile.name}
                      </p>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          <button
            onClick={calculateRisk}
            className="calculate-button"
          >
            Calculate
          </button>
        </div>

        {/* Results */}
        <div className="results-panel">
          <div className="results-container">
            <h2 className="results-title">Predicted Clinical Outcomes:</h2>
            
            <div className="outcomes-container">
              <div className="outcome-row">
                <span className="outcome-label">30-day <span className="mortality">mortality</span>:</span>
                <input
                  type="text"
                  className="result-input"
                  value={results.mortality30}
                  readOnly
                />
              </div>
              
              <div className="outcome-row">
                <span className="outcome-label">30-day <span className="readmission">readmission</span>:</span>
                <input
                  type="text"
                  className="result-input"
                  value={results.readmission30}
                  readOnly
                />
              </div>
            </div>
          </div>

          <div className="description">
            <p className="description-text">
              This calculator predicts 30-day mortality and 30-day readmission, incorporating prior
              diagnoses and other important patient data.
            </p>
            <p className="disclaimer-italic">
              **Disclaimer: This tool is for educational and clinical decision support only. Always use clinical judgment and consult appropriate healthcare providers.**
            </p>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="footer">
        <p>Questions or comments? <a href="mailto:contact@brown.com" className="footer-link">Email Us</a>.</p>
      </div>
    </div>
  );
};

export default OutcomeCalculator;