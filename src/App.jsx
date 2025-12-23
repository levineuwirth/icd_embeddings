import React, { useState } from 'react';
import axios from 'axios';

const OutcomeCalculator = () => {
  const API_URL = import.meta.env.VITE_API_BASE_URL;
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    primaryPayer: '',
    householdIncome: '',
    icdMethod: 'manual', // user has choice between 'manual', 'paste', or 'upload'
    icdCodes: ['', '', '', '', ''],
    uploadedFile: null,
    pastedText: ''
  });

  const [results, setResults] = useState({
    mortality30: '',
    readmission30: ''
  });

  const [icdSearchResults, setIcdSearchResults] = useState([]);
  const [validationResults, setValidationResults] = useState(null);
  const [ageError, setAgeError] = useState('');
  const [ageWarning, setAgeWarning] = useState('');
  const [isCalculating, setIsCalculating] = useState(false);

  const primaryPayerOptions = [
    'Medicare',
    'Medicaid',
    'Private Insurance',
    'Self-Pay',
    'Workers Compensation',
    'Other'
  ];

  const validateAge = (age) => {
    const ageNum = parseInt(age);

    if (age === '' || isNaN(ageNum)) {
      setAgeError('');
      setAgeWarning('');
      return { valid: false, error: '' };
    }

    if (ageNum < 0) {
      setAgeError('Age cannot be less than 0.');
      setAgeWarning('');
      return { valid: false, error: 'Age cannot be less than 0.' };
    }

    if (ageNum >= 125) {
      setAgeError('Age cannot be 125 or greater.');
      setAgeWarning('');
      return { valid: false, error: 'Age cannot be 125 or greater.' };
    }

    if (ageNum >= 90 && ageNum <= 124) {
      setAgeError('');
      setAgeWarning('Ages 90-124 will be submitted as 90 (dataset constraint).');
      return { valid: true, adjustedAge: 90 };
    }

    setAgeError('');
    setAgeWarning('');
    return { valid: true, adjustedAge: ageNum };
  };

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));

    // Validate age whenever it changes
    if (field === 'age') {
      validateAge(value);
    }
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

    const uploadFormData = new FormData();
    uploadFormData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/upload_icd_file/`, uploadFormData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      const { valid_codes, invalid_codes, warnings } = response.data;
      setValidationResults(response.data);
      setFormData(prev => ({
        ...prev,
        icdCodes: valid_codes
      }));
    } catch (error) {
      console.error("Error uploading file:", error);
      alert("There was an error processing the uploaded file.");
    }
  };

  const handlePasteChange = (value) => {
    setFormData(prev => ({
      ...prev,
      pastedText: value
    }));
  };

  const handleParsePastedCodes = async () => {
    if (!formData.pastedText.trim()) {
      alert("Please paste some ICD codes first.");
      return;
    }

    try {
      const response = await axios.post(`${API_URL}/parse_icd_codes/`, {
        text: formData.pastedText
      });

      const { valid_codes, invalid_codes, warnings } = response.data;
      setValidationResults(response.data);
      setFormData(prev => ({
        ...prev,
        icdCodes: valid_codes
      }));
    } catch (error) {
      console.error("Error parsing codes:", error);
      alert("There was an error parsing the pasted codes.");
    }
  };

  const calculateRisk = async () => {
    const { age, gender, primaryPayer, householdIncome, icdCodes } = formData;

    // Build payload with optional demographic fields
    const payload = {
      icd_codes: icdCodes.filter(code => code.trim() !== '')
    };

    // Validate and add age if provided
    if (age && age.trim() !== '') {
      const ageValidation = validateAge(age);
      if (!ageValidation.valid) {
        if (ageValidation.error) {
          alert(`Invalid age: ${ageValidation.error}`);
        } else {
          alert('Please enter a valid age.');
        }
        return;
      }
      payload.age = ageValidation.adjustedAge;
    }

    // Add gender if provided
    if (gender) {
      payload.female = gender === 'F' ? 1 : 0;
    }

    // Add primary payer if provided
    if (primaryPayer) {
      const pay1Mapping = {
        'Medicare': 1,
        'Medicaid': 2,
        'Private Insurance': 3,
        'Self-Pay': 4,
        'Workers Compensation': 5,
        'Other': 6
      };
      payload.pay1 = pay1Mapping[primaryPayer];
    }

    // Add household income if provided
    if (householdIncome) {
      payload.zipinc_qrtl = parseInt(householdIncome);
    }

    // Check if ICD codes are provided
    if (payload.icd_codes.length === 0) {
      alert('Please provide at least one valid ICD code.');
      return;
    }

    setIsCalculating(true);

    try {
      const response = await axios.post(`${API_URL}/predict_flex/`, payload);
      const { readmission, mortality } = response.data;

      setResults({
        readmission30: `${(readmission.prediction * 100).toFixed(1)}%`,
        mortality30: `${(mortality.prediction * 100).toFixed(1)}%`
      });
    } catch (error) {
      console.error("Error calculating risk:", error);
      if (error.response && error.response.data && error.response.data.detail) {
        alert(`Error: ${error.response.data.detail}`);
      } else {
        alert("There was an error calculating the risk. Please check the inputs and try again.");
      }
    } finally {
      setIsCalculating(false);
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
          <p style={{ fontSize: '0.875rem', color: '#666', marginBottom: '1rem' }}>
            All demographic fields are optional. For best accuracy, provide all fields. ICD codes are required.
          </p>

          <div className="form-container">
            {/* Age */}
            <div className="form-row">
              <label className="form-label">Age <span style={{ fontSize: '0.75rem', color: '#999' }}>(optional)</span>:</label>
              <div style={{ flex: 1 }}>
                <input
                  type="number"
                  placeholder="years"
                  className="form-input"
                  value={formData.age}
                  onChange={(e) => handleInputChange('age', e.target.value)}
                  style={ageError ? { borderColor: 'red' } : {}}
                />
                {ageError && (
                  <div style={{ color: 'red', fontSize: '0.875rem', marginTop: '0.25rem' }}>
                    {ageError}
                  </div>
                )}
                {ageWarning && !ageError && (
                  <div style={{ color: 'orange', fontSize: '0.875rem', marginTop: '0.25rem' }}>
                    {ageWarning}
                  </div>
                )}
              </div>
            </div>

            {/* Gender */}
            <div className="form-row">
              <label className="form-label">Gender <span style={{ fontSize: '0.75rem', color: '#999' }}>(optional)</span>:</label>
              <div className="button-group">
                <button
                  className={`toggle-button ${formData.gender === 'M' ? 'active' : ''}`}
                  onClick={() => handleInputChange('gender', formData.gender === 'M' ? '' : 'M')}
                >
                  M
                </button>
                <button
                  className={`toggle-button ${formData.gender === 'F' ? 'active' : ''}`}
                  onClick={() => handleInputChange('gender', formData.gender === 'F' ? '' : 'F')}
                >
                  F
                </button>
              </div>
            </div>

            {/* Expected Primary Payer */}
            <div className="form-row">
              <label className="form-label">Expected Primary Payer <span style={{ fontSize: '0.75rem', color: '#999' }}>(optional)</span>:</label>
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
              <label className="form-label">Household Income Quartile <span style={{ fontSize: '0.75rem', color: '#999' }}>(optional)</span>:</label>
              <div className="button-group">
                {[1, 2, 3, 4].map(quartile => (
                  <button
                    key={quartile}
                    className={`toggle-button ${formData.householdIncome === quartile.toString() ? 'active' : ''}`}
                    onClick={() => handleInputChange('householdIncome', formData.householdIncome === quartile.toString() ? '' : quartile.toString())}
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
                      onChange={() => {
                        handleInputChange('icdMethod', 'manual');
                        setValidationResults(null);
                      }}
                    />
                    Manual Input
                  </label>
                  <label className="radio-label">
                    <input
                      type="radio"
                      name="icdMethod"
                      checked={formData.icdMethod === 'paste'}
                      onChange={() => {
                        handleInputChange('icdMethod', 'paste');
                        setValidationResults(null);
                      }}
                    />
                    Paste Codes
                  </label>
                  <label className="radio-label">
                    <input
                      type="radio"
                      name="icdMethod"
                      checked={formData.icdMethod === 'upload'}
                      onChange={() => {
                        handleInputChange('icdMethod', 'upload');
                        setValidationResults(null);
                      }}
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
                ) : formData.icdMethod === 'paste' ? (
                  <div className="paste-container">
                    <p className="instruction-text">
                      Paste ICD codes in any format (comma, space, or line-separated):
                    </p>
                    <textarea
                      className="paste-textarea"
                      placeholder="Example: I10, E11.9, J44.0&#10;or one per line:&#10;I10&#10;E11.9&#10;J44.0"
                      value={formData.pastedText}
                      onChange={(e) => handlePasteChange(e.target.value)}
                      rows={6}
                    />
                    <button
                      onClick={handleParsePastedCodes}
                      className="parse-button"
                    >
                      Parse & Validate Codes
                    </button>
                  </div>
                ) : (
                  <div className="upload-container">
                    <p className="instruction-text">
                      Upload a .txt or .csv file with ICD codes (any format accepted)
                    </p>
                    <input
                      type="file"
                      accept=".csv,.txt"
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

                {/* Validation Results */}
                {validationResults && (
                  <div className="validation-results">
                    <div className="validation-summary">
                      <span className="valid-count">✓ {validationResults.valid_codes.length} valid codes</span>
                      {validationResults.invalid_codes.length > 0 && (
                        <span className="invalid-count">⚠ {validationResults.invalid_codes.length} invalid codes</span>
                      )}
                    </div>

                    {validationResults.warnings.length > 0 && (
                      <div className="validation-warnings">
                        {validationResults.warnings.map((warning, idx) => (
                          <p key={idx} className="warning-text">⚠ {warning}</p>
                        ))}
                      </div>
                    )}

                    {validationResults.invalid_codes.length > 0 && (
                      <div className="invalid-codes-list">
                        <p className="invalid-header">Invalid codes found:</p>
                        {validationResults.invalid_codes.map((item, idx) => (
                          <div key={idx} className="invalid-code-item">
                            <span className="invalid-code-name">{item.code}</span>
                            {item.suggestions.length > 0 && (
                              <span className="suggestions">
                                (Did you mean: {item.suggestions.join(', ')}?)
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    )}

                    {validationResults.valid_codes.length > 0 && (
                      <div className="valid-codes-preview">
                        <p className="preview-header">Valid codes loaded:</p>
                        <p className="codes-preview">{validationResults.valid_codes.join(', ')}</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          <button
            onClick={calculateRisk}
            className="calculate-button"
            disabled={isCalculating}
          >
            {isCalculating ? 'Calculating...' : 'Calculate'}
          </button>
        </div>

        {/* Results */}
        <div className="results-panel">
          <div className="results-container">
            <h2 className="results-title">Predicted Clinical Outcomes:</h2>

            {isCalculating ? (
              <div style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                padding: '3rem',
                fontSize: '1.125rem',
                color: '#666'
              }}>
                Calculating predictions...
              </div>
            ) : (
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
            )}
          </div>

          <div className="description">
            <p className="description-text">
              This calculator predicts 30-day mortality and 30-day readmission risk using advanced
              machine learning models trained on ICD-10 diagnosis codes and patient demographics.
            </p>
            <p className="disclaimer-italic">
              **Disclaimer: This tool is for educational and clinical decision support only. Always use clinical judgment and consult appropriate healthcare providers.**
            </p>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="footer">
        <p>Questions or comments? <a href="mailto:levi_neuwirth@brown.edu" className="footer-link">Email Us</a>.</p>
      </div>
    </div>
  );
};

export default OutcomeCalculator;