// Smooth Scrolling on Navigation Click
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Highlight Active Navigation Link While Scrolling
window.addEventListener('scroll', () => {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-links a');

    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop - 100;
        if (scrollY >= sectionTop) {
            current = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// Image Upload & Prediction Logic
const imageInput = document.getElementById('imageInput');
const fileInfo = document.getElementById('fileInfo');
const predictBtn = document.getElementById('predictBtn');
const imagePreview = document.getElementById('imagePreview');
const resultContainer = document.getElementById('resultContainer');
const classificationResult = document.getElementById('classificationResult');
const uploadArea = document.querySelector('.upload-area');

let selectedFile = null;

// Drag & Drop Upload Support
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        handleFileSelect(files[0]);
    }
});

// File Input Change
imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Handle File Select
function handleFileSelect(file) {
    selectedFile = file;
    fileInfo.textContent = file.name;
    predictBtn.disabled = false;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.innerHTML = `<img src="${e.target.result}" alt="Preview" class="preview-image">`;
    };
    reader.readAsDataURL(file);

   
    resultContainer.style.display = 'none';
}


predictBtn.addEventListener('click', () => {
    if (!selectedFile) return;

    predictBtn.disabled = true;
    predictBtn.textContent = 'Analyzing...';

    const formData = new FormData();
    formData.append("image", selectedFile);

    fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        classificationResult.innerHTML = `
            <div style="margin-top: 15px; font-size: 20px; font-weight: bold; color: ${
                data.class.toLowerCase().includes('healthy') ? '#27ae60' : '#e74c3c'
            };">
                Test Class: ${data.class} (${Math.round(data.confidence * 100)}%)
            </div>
        `;

        resultContainer.style.display = 'block';
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });

        predictBtn.disabled = false;
        predictBtn.textContent = 'Predict';
    })
    .catch(error => {
        classificationResult.innerHTML =` <div style="color:red;" >Prediction failed. Please try again.</div>`;
        resultContainer.style.display = 'block';
        predictBtn.disabled = false;
        predictBtn.textContent = 'Predict';
        console.error('Prediction error:', error);
    });
});
