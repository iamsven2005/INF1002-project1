(function () {
    var addButton = document.getElementById('add-pet');
    var rowsContainer = document.getElementById('pet-rows');
    var template = document.getElementById('pet-row-template');

    if (!addButton || !rowsContainer || !template) {
        return;
    }

    function clearRow(row) {
        var fields = row.querySelectorAll('input');
        fields.forEach(function (field) {
            if (field.type === 'file') {
                field.value = '';
                return;
            }
            field.value = '';
        });

        var photo = row.querySelector('img');
        if (photo) {
            photo.remove();
        }
        var note = row.querySelector('.text-secondary.small');
        if (!note) {
            var col = row.querySelector('.col-md-4.d-flex');
            if (col) {
                note = document.createElement('span');
                note.className = 'text-secondary small';
                note.textContent = 'No photo selected';
                col.insertBefore(note, col.firstChild);
            }
        }
    }

    function attachRemoveHandlers() {
        var buttons = rowsContainer.querySelectorAll('.remove-pet');
        buttons.forEach(function (button) {
            if (button.dataset.bound === 'true') {
                return;
            }
            button.dataset.bound = 'true';

            button.addEventListener('click', function () {
                var row = button.closest('.pet-row');
                if (!row) {
                    return;
                }

                var allRows = rowsContainer.querySelectorAll('.pet-row');
                if (allRows.length <= 1) {
                    clearRow(row);
                    return;
                }

                row.remove();
            });
        });
    }

    addButton.addEventListener('click', function () {
        var clone = template.content.cloneNode(true);
        rowsContainer.appendChild(clone);
        attachRemoveHandlers();
    });

    attachRemoveHandlers();
})();
