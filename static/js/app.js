
function showLoading() {
    const form = document.querySelector('form');
    const fileInput = document.getElementById('formFile');
    const selectInput = document.getElementById('floatingSelect');
    
    if (!fileInput.files.length || !selectInput.value) {
        return false;
    }
    
    document.getElementById("loadingOverlay").style.display = "flex";
    return true;
}
