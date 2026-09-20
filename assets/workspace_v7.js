// Capture before the legacy keyboard listener. Only applies to the v7 editor.
document.addEventListener('keydown', function(event) {
    if (!document.getElementById('ws-fitting')) return;
    const target = event.target;
    // React Select needs its navigation keys; typed shortcut letters stay local.
    if (target && target.closest('#ws-profile, #ws-catalog') &&
        ['ArrowDown', 'ArrowUp', 'Enter', 'Escape', 'Tab'].includes(event.key)) return;
    if (target && (target.tagName === 'TEXTAREA' || target.tagName === 'INPUT' || target.isContentEditable)) {
        event.stopPropagation();
    }
}, true);
window.addEventListener('beforeunload', function(event) {
    if (window.assignerEditorDirty) {
        event.preventDefault();
        event.returnValue = '';
    }
});
document.addEventListener('click', function(event) {
    if (event.target.closest('#ws-reload') && window.assignerEditorDirty &&
        !window.confirm('Discard the current file draft and reload it from disk? Other file drafts are retained.')) {
        event.stopImmediatePropagation();
        event.preventDefault();
    }
}, true);
function updateAssignerLines() {
    const editor = document.getElementById('ws-editor');
    const gutter = document.getElementById('ws-line-numbers');
    if (!editor || !gutter) return;
    const count = editor.value.split('\n').length;
    if (gutter.dataset.count !== String(count)) {
        gutter.textContent = Array.from({length: count}, (_, i) => i + 1).join('\n');
        gutter.dataset.count = String(count);
    }
    gutter.scrollTop = editor.scrollTop;
}
document.addEventListener('input', updateAssignerLines, true);
document.addEventListener('scroll', updateAssignerLines, true);
setInterval(updateAssignerLines, 600);
