window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        updateFornaContainer: function (tableData, selectedRowIds, show_mfe, sequence, mfe) {
            if (!tableData  || !sequence) {
                return window.dash_clientside.no_update;
            }

            const length = sequence.length;
            let sequences = [];
            if (selectedRowIds) {
                sequences = selectedRowIds.map(id => {
                    const row = tableData.find(r => r.id === id);
                    return {
                        sequence: sequence, structure: row.Structure
                    };
                });

            }

            if (show_mfe) {
                sequences.push({sequence: sequence, structure: mfe});
            }


            return sequences;
        },
        downloadStructureTable: function(n_clicks, data, columns, seqid) {
            if (!n_clicks || !data || !columns) {
                return;
            }

            let tsv = '';
            const headers = columns.map(col => col.name).join('\t');
            tsv += headers + '\n';

            for (const row of data) {
                const line = columns.map(col => row[col.id]).join('\t');
                tsv += line + '\n';
            }

            const blob = new Blob([tsv], { type: 'text/tab-separated-values' });
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = 'sampled_structures_' + seqid + '.tsv';
            a.click();
            URL.revokeObjectURL(url);
        },
        switchLayout: function (mode, fig, lightLayout, darkLayout) {
            let layout = mode ? lightLayout : darkLayout;
            return {
                ...fig, layout: {...fig.layout, ...layout}
            };
        },
        highlightNucleotides: function(nt_i, nt_j) {
            if (nt_i == null && nt_j == null) {
                return {};
            }

            let colorValues = {};
            if (nt_i != null) {
                colorValues[(nt_i).toString()] = "var(--bs-secondary)";
            }
            if (nt_j != null) {
                colorValues[(nt_j).toString()] = "var(--bs-primary)";
            }

            return {
                colorValues: {
                    "": colorValues  // "" means apply to all molecules unless overridden
                }
            };
        }
    }
});


function resizeFornaBackgroundRect() {
    const observer = new MutationObserver((mutationsList, observer) => {
        const rect = document.querySelector('svg .background');
        if (rect) {
            rect.setAttribute('width', '100%');
            console.log('[forna] Resized background <rect> to 100%');
            observer.disconnect(); // stop observing once it's done
        }
    });

    // Start observing the entire document body for child changes
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
}

// Run when the page is fully loaded (after Dash renders initial layout)
window.addEventListener('load', function () {
    resizeFornaBackgroundRect();
});