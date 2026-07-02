/**
 * Ames Housing Price Predictor - Frontend JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const predictBtn = document.getElementById('predict-btn');
    const resultSection = document.getElementById('result-section');
    const errorSection = document.getElementById('error-section');
    const errorMessage = document.getElementById('error-message');
    const predictionResult = document.getElementById('prediction-result');
    const confidenceSpan = document.getElementById('confidence');
    
    // Add loading state
    function setLoading(isLoading) {
        if (isLoading) {
            predictBtn.disabled = true;
            predictBtn.innerHTML = '<span class="loading-spinner"></span> Predicting...';
        } else {
            predictBtn.disabled = false;
            predictBtn.innerHTML = '🔮 Predict Price';
        }
    }
    
    // Show error message
    function showError(message) {
        errorMessage.textContent = message;
        errorSection.style.display = 'block';
        resultSection.style.display = 'none';
        setTimeout(() => {
            errorSection.style.display = 'none';
        }, 5000);
    }
    
    // Handle form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Hide previous results
        resultSection.style.display = 'none';
        errorSection.style.display = 'none';
        
        // Show loading
        setLoading(true);
        
        try {
            // Collect form data
            const formData = {
                'Overall Qual': parseInt(document.getElementById('overall_qual').value),
                'Overall Cond': parseInt(document.getElementById('overall_cond').value),
                'Gr Liv Area': parseInt(document.getElementById('gr_liv_area').value),
                'Year Built': parseInt(document.getElementById('year_built').value),
                'Year Remod/Add': parseInt(document.getElementById('year_remod').value),
                'Yr Sold': parseInt(document.getElementById('year_sold').value),
                'Bedroom AbvGr': parseInt(document.getElementById('bedrooms').value),
                'Total Bsmt SF': parseInt(document.getElementById('total_bsmt_sf').value),
                'Fireplaces': parseInt(document.getElementById('fireplaces').value),
                'Garage Area': parseInt(document.getElementById('garage_area').value),
                'Full Bath': parseInt(document.getElementById('full_bath').value),
                'TotRms AbvGrd': parseInt(document.getElementById('tot_rms').value),
                'Lot Area': parseInt(document.getElementById('lot_area').value),
                'Neighborhood': document.getElementById('neighborhood').value,
                'Sale Condition': document.getElementById('sale_condition').value
            };
            
            // Validate input
            for (const [key, value] of Object.entries(formData)) {
                if (isNaN(value) && typeof value === 'number') {
                    showError(`Please enter a valid value for ${key}`);
                    setLoading(false);
                    return;
                }
            }
            
            // Send prediction request
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Show prediction
                predictionResult.textContent = data.formatted_price || `$${data.prediction.toLocaleString()}`;
                confidenceSpan.textContent = data.confidence ? `${(data.confidence * 100).toFixed(1)}%` : '95%';
                resultSection.style.display = 'block';
                
                // Animate result
                predictionResult.style.animation = 'none';
                setTimeout(() => {
                    predictionResult.style.animation = 'fadeInUp 0.5s ease';
                }, 10);
            } else {
                showError(data.error || 'An error occurred while making the prediction.');
            }
        } catch (error) {
            showError('Network error: Could not connect to the server. Please try again.');
            console.error('Error:', error);
        } finally {
            setLoading(false);
        }
    });
    
    // Add real-time validation for number inputs
    document.querySelectorAll('input[type="number"]').forEach(input => {
        input.addEventListener('input', function() {
            const min = parseInt(this.min);
            const max = parseInt(this.max);
            const value = parseInt(this.value);
            
            if (value < min) {
                this.value = min;
            } else if (value > max) {
                this.value = max;
            }
        });
    });
    
    // Check server health on load
    async function checkHealth() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            if (data.status === 'healthy') {
                console.log('Server is healthy');
            } else {
                console.warn('Server health check failed:', data);
            }
        } catch (error) {
            console.warn('Could not reach server:', error);
        }
    }
    
    checkHealth();
});