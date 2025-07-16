window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        updateFornaContainer: function (tableData, selectedRowIds, sequence) {
            if (!tableData || !selectedRowIds || !sequence) {
                return window.dash_clientside.no_update;
            }

            const length = sequence.length;

            const sequences = selectedRowIds.map(id => {
                const row = tableData.find(r => r.id === id);
                return {
                    sequence: sequence, structure: row.Structure
                };
            });
            console.log(sequences);


            return sequences;
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
                colorValues[(nt_i +1).toString()] = 'red';
            }
            if (nt_j != null) {
                colorValues[(nt_j+1).toString()] = 'red';
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