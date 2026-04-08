let timeout = null;

document.getElementById('search-input').addEventListener('input', function() {
    const keyword = this.value.trim();
    const suggestionBox = document.getElementById('suggestions');

    clearTimeout(timeout);

    if (keyword.length === 0) {
        suggestionBox.style.display = 'none';
        return;
    }

    timeout = setTimeout(() => {
        fetch(`/get_suggestions?keyword=${encodeURIComponent(keyword)}`)
            .then(response => response.json())
            .then(data => {
                if (data.length > 0) {
                    suggestionBox.innerHTML = data.map(item => 
                        `<div class="suggestion-item">${item.ten_hang}</div>`
                    ).join('');
                    suggestionBox.style.display = 'block';
                } else {
                    suggestionBox.style.display = 'none';
                }
            });
    }, 300);
});
document.getElementById('suggestions').addEventListener('click', function(e) {
    if (e.target.classList.contains('suggestion-item')) {
        document.getElementById('search-input').value = e.target.innerText;
        this.style.display = 'none';
    }
});