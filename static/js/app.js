/**
 * Nutrilens AI - Frontend Application
 * Main JavaScript Logic
 */

// ==========================================
// STATE MANAGEMENT
// ==========================================

const state = {
    selectedFile: null,
    imageData: {
        width: 0,
        height: 0,
        filename: '',
        size: 0
    },
    analysisResult: null,
    startTime: 0,
    currentStep: 0,
    steps: [
        { name: 'Detection', description: 'Detecting food and plate regions' },
        { name: 'Extraction', description: 'Extracting ingredient information' },
        { name: 'Segmentation', description: 'Segmenting individual ingredients' },
        { name: 'Depth Estimation', description: 'Estimating depth map' },
        { name: 'Geometry', description: 'Computing volume and geometry' },
        { name: 'Nutrition', description: 'Calculating nutritional values' }
    ]
};

// ==========================================
// CONVERSION UTILITIES
// ==========================================

function convertHeight(value, unit) {
    if (unit === 'mm') return value / 10; // mm to cm
    return value; // already in cm
}

function convertArea(value, unit) {
    if (unit === 'mm2') return value / 100; // mm² to cm²
    return value; // already in cm²
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// ==========================================
// FILE HANDLING
// ==========================================

document.getElementById('imageInput').addEventListener('change', (e) => {
    state.selectedFile = e.target.files[0];
    if (state.selectedFile) {
        document.getElementById('processBtn').disabled = false;
    }
});

function previewImage() {
    if (!state.selectedFile) {
        showAlert('error', 'Please select an image first');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            state.imageData.width = img.width;
            state.imageData.height = img.height;
            state.imageData.filename = state.selectedFile.name;
            state.imageData.size = state.selectedFile.size;

            // Display preview
            document.getElementById('imagePreview').src = e.target.result;
            document.getElementById('imagePreview').style.display = 'block';

            // Display info
            document.getElementById('previewFilename').textContent = state.imageData.filename;
            document.getElementById('previewDimensions').textContent = 
                `${state.imageData.width} × ${state.imageData.height} px`;
            document.getElementById('previewSize').textContent = 
                formatFileSize(state.imageData.size);
            document.getElementById('previewResolution').textContent = 
                `${Math.round(Math.sqrt(state.imageData.width * state.imageData.height))} PPI`;
            document.getElementById('previewInfo').style.display = 'block';

            showAlert('success', 'Image preview loaded successfully!');
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(state.selectedFile);
}

// ==========================================
// ALERT SYSTEM
// ==========================================

function showAlert(type, message) {
    const alertElement = document.getElementById(`alert${type.charAt(0).toUpperCase() + type.slice(1)}`);
    const textElement = alertElement.querySelector('p');
    textElement.textContent = message;
    alertElement.classList.add('active');
    
    setTimeout(() => {
        alertElement.classList.remove('active');
    }, 5000);
}

// ==========================================
// PROGRESS TRACKING
// ==========================================

function initializeProgressBar() {
    state.currentStep = 0;
    document.getElementById('progressSection').classList.add('active');
    document.getElementById('stepsLog').classList.add('active');
    document.getElementById('stepsLog').innerHTML = '';
    
    // Add all steps to the log
    state.steps.forEach((step, index) => {
        const stepEl = createStepElement(step, index === 0 ? 'processing' : 'pending');
        document.getElementById('stepsLog').appendChild(stepEl);
    });
}

function createStepElement(step, status) {
    const div = document.createElement('div');
    div.className = 'step-item';
    div.innerHTML = `
        <div class="step-icon ${status}">${status === 'processing' ? '⟳' : status === 'done' ? '✓' : status === 'error' ? '✕' : '◯'}</div>
        <div class="step-text">
            <div class="step-name">${step.name}</div>
            <div class="step-description">${step.description}</div>
        </div>
        <div class="step-time" id="stepTime${status}">0.0s</div>
    `;
    return div;
}

function updateProgressBar(percentage, stepName) {
    document.getElementById('progressBar').style.width = percentage + '%';
    document.getElementById('stepName').textContent = stepName;
    const elapsed = ((Date.now() - state.startTime) / 1000).toFixed(1);
    document.getElementById('stepTime').textContent = elapsed + 's';
}

function completeStep(stepIndex) {
    const stepItems = document.querySelectorAll('.step-item');
    if (stepItems[stepIndex]) {
        const stepItem = stepItems[stepIndex];
        const icon = stepItem.querySelector('.step-icon');
        icon.textContent = '✓';
        icon.className = 'step-icon done';
    }
}

function nextStep(stepIndex) {
    const stepItems = document.querySelectorAll('.step-item');
    if (stepItems[stepIndex]) {
        const stepItem = stepItems[stepIndex];
        const icon = stepItem.querySelector('.step-icon');
        icon.textContent = '⟳';
        icon.className = 'step-icon processing';
    }
}

// ==========================================
// API COMMUNICATION
// ==========================================

async function processImage() {
    if (!state.selectedFile) {
        showAlert('error', 'Please select an image first');
        return;
    }

    // Disable button and show loading
    document.getElementById('processBtn').disabled = true;
    document.getElementById('loadingContainer').classList.add('active');
    initializeProgressBar();

    state.startTime = Date.now();

    try {
        // Get reference parameters
        const cameraHeight = convertHeight(
            parseFloat(document.getElementById('cameraHeight').value),
            document.getElementById('heightUnit').value
        );
        const pixelArea = convertArea(
            parseFloat(document.getElementById('pixelArea').value),
            document.getElementById('areaUnit').value
        );

        // Create FormData
        const formData = new FormData();
        formData.append('file', state.selectedFile);
        formData.append('camera_height_ref', cameraHeight);
        formData.append('pixel_area_ref', pixelArea);

        // Simulate progress updates
        const progressInterval = setInterval(() => {
            const elapsed = (Date.now() - state.startTime) / 1000;
            const progress = Math.min(85, (elapsed / 60) * 100);
            updateProgressBar(progress, 'Processing...');
        }, 500);

        // Update steps progressively
        let stepIndex = 0;
        const stepInterval = setInterval(() => {
            if (stepIndex < state.steps.length - 1) {
                completeStep(stepIndex);
                stepIndex++;
                if (stepIndex < state.steps.length) {
                    nextStep(stepIndex);
                    updateProgressBar((stepIndex / state.steps.length) * 85, state.steps[stepIndex].name);
                }
            }
        }, 8000); // Each step takes ~8 seconds

        // Make API request
        const response = await fetch('/api/v1/nutrition/analyze_debug', {
            method: 'POST',
            body: formData
        });

        clearInterval(progressInterval);
        clearInterval(stepInterval);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail?.message || 'Analysis failed');
        }

        const result = await response.json();
        state.analysisResult = result;

        // Complete all steps
        state.steps.forEach((_, i) => completeStep(i));
        updateProgressBar(100, 'Analysis Complete!');

        // Show results
        setTimeout(() => {
            displayResults(result);
            document.getElementById('loadingContainer').classList.remove('active');
            document.getElementById('progressSection').classList.remove('active');
            document.getElementById('stepsLog').classList.remove('active');
            document.getElementById('resultsSection').classList.add('active');
            
            const totalTime = ((Date.now() - state.startTime) / 1000).toFixed(2);
            document.getElementById('processingTime').textContent = totalTime + 's';
            document.getElementById('deviceInfo').textContent = result.device;

            showAlert('success', 'Analysis completed successfully!');
        }, 1000);

    } catch (error) {
        console.error('Error:', error);
        showAlert('error', error.message || 'An error occurred during analysis');
        document.getElementById('loadingContainer').classList.remove('active');
    } finally {
        document.getElementById('processBtn').disabled = false;
    }
}

// ==========================================
// RESULTS DISPLAY
// ==========================================

function displayResults(result) {
    // Display Summary
    displaySummary(result);
    // Display Details
    displayDetails(result);
    // Display Debug Info
    if (result.debug_info && result.debug_info.images) {
        displayDebugImages(result.debug_info.images);
    }
    // Display Raw Data
    displayRawData(result);
}

function displaySummary(result) {
    const summaryGrid = document.getElementById('summaryGrid');
    summaryGrid.innerHTML = '';

    const summary = result.summary;
    const metrics = [
        { label: 'Total Mass', value: summary.total_mass_g, unit: 'g' },
        { label: 'Total Calories', value: summary.total_calories_kcal, unit: 'kcal' },
        { label: 'Protein', value: summary.total_protein_g, unit: 'g' },
        { label: 'Fat', value: summary.total_fat_g, unit: 'g' },
        { label: 'Carbs', value: summary.total_carbs_g, unit: 'g' }
    ];

    metrics.forEach(metric => {
        const card = document.createElement('div');
        card.className = 'summary-card';
        card.innerHTML = `
            <h3>${metric.label}</h3>
            <div class="value">${metric.value.toFixed(2)}<span class="unit">${metric.unit}</span></div>
        `;
        summaryGrid.appendChild(card);
    });
}

function displayDetails(result) {
    const tbody = document.getElementById('detailsBody');
    tbody.innerHTML = '';

    result.items.forEach(item => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${item.ingredient}</td>
            <td><strong>${item.matched_name}</strong></td>
            <td>${item.mass_g.toFixed(2)}</td>
            <td>${item.calories_kcal.toFixed(2)}</td>
            <td>${item.protein_g.toFixed(2)}</td>
            <td>${item.fat_g.toFixed(2)}</td>
            <td>${item.carbs_g.toFixed(2)}</td>
            <td>${(item.confidence * 100).toFixed(1)}%</td>
        `;
        tbody.appendChild(row);
    });
}

function displayDebugImages(images) {
    const gallery = document.getElementById('debugGallery');
    gallery.innerHTML = '';

    Object.entries(images).forEach(([key, base64]) => {
        const card = document.createElement('div');
        card.className = 'debug-image-card';
        card.innerHTML = `
            <div class="debug-image-container">
                <img src="data:image/png;base64,${base64}" alt="${key}">
            </div>
            <div class="debug-image-label">${key.replace(/_/g, ' ')}</div>
        `;
        gallery.appendChild(card);
    });
}

function displayRawData(result) {
    const rawPre = document.getElementById('rawDataPre');
    rawPre.textContent = JSON.stringify(result, null, 2);
}

// ==========================================
// TAB SWITCHING
// ==========================================

function switchTab(tabName) {
    // Remove active class from all tabs
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });

    // Add active class to selected tab
    event.target.classList.add('active');
    document.getElementById(tabName + 'Tab').classList.add('active');
}

// ==========================================
// INITIALIZATION
// ==========================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('Nutrilens AI Frontend loaded');
    
    // Set initial device info
    document.getElementById('deviceInfo').textContent = 'Loading...';
});
