/* This file contains most of the code that manages the details of a
 * qurro visualization.
 *
 * RRVDisplay.makeRankPlot() and RRVDisplay.makeSamplePlot() were based on the
 * Basic Example in https://github.com/vega/vega-embed/.
 */
define(["./feature_computation", "./dom_utils", "vega", "vega-embed"], function(
    feature_computation,
    dom_utils,
    vega,
    vegaEmbed
) {
    class RRVDisplay {
        /* Class representing a display in qurro (involving two plots:
         * one bar plot containing feature ranks, and one scatterplot
         * describing sample log-ratios of feature abundances). These plots are
         * referred to in this code as the "rank" and "sample" plot,
         * respectively.
         *
         * Its constructor takes as arguments JSON objects representing the rank
         * and sample plot in Vega-Lite, as well as a JSON object containing
         * count information for the input feature table (these should all be
         * generated and moved into the main.js file for this visualization
         * by Qurro's python code).
         *
         * This class assumes that some DOM elements exist on the page in
         * order to function properly. The most notable of these are two
         * <div> elements -- one with the ID "rankPlot" and one with the ID
         * "samplePlot" -- in which Vega visualizations of these plots will be
         * embedded using Vega-Embed.
         *
         * (It'd be possible in the future to make the IDs used to find these
         * DOM elements configurable in the class constructor, but I don't
         * think that would be super useful unless you want to embed
         * qurro' web interface in a bunch of other environments.)
         *
         * You need to call this.makePlots() to actually make this interactive
         * / show things.
         */
        constructor(rankPlotJSON, samplePlotJSON, countJSON) {
            // Used for selections of log-ratios between single features (via
            // the rank plot)
            this.onHigh = true;
            this.newFeatureLow = undefined;
            this.newFeatureHigh = undefined;
            this.oldFeatureLow = undefined;
            this.oldFeatureHigh = undefined;

            // For selections of potentially many features (not via the rank plot)
            this.topFeatures = undefined;
            this.botFeatures = undefined;

            // Used when looking up a feature's count.
            this.featureCts = countJSON;
            // Used when searching through features.
            // Since we filtered out empty features in the python side of
            // things, we know that every feature should be represented in the
            // count JSON's keys.
            this.featureIDs = Object.keys(this.featureCts);

            // Just a list of all sample IDs.
            this.sampleIDs = RRVDisplay.identifySampleIDs(samplePlotJSON);
            // Used when letting the user know how many samples are present in
            // the sample plot.
            this.sampleCount = this.sampleIDs.length;

            // a mapping from "reason" (i.e. "balance", "xAxis", "color") to
            // list of dropped sample IDs.
            //
            // "balance" maps to this.sampleIDs right now because all samples
            // have a null balance starting off.
            //
            // NOTE: xAxis and color might already exclude some samples from
            // being shown in the default categorical encoding. Their
            // corresponding lists will be updated in makeSamplePlot(), before
            // dom_utils.updateMainSampleShownDiv() will be called (so the
            // nulls will be replaced with actual lists).
            this.droppedSamples = {
                balance: this.sampleIDs,
                xAxis: null,
                color: null
            };

            // Set when the sample plot JSON is loaded. Used to populate
            // possible sample plot x-axis/colorization options.
            this.metadataCols = undefined;

            // Ordered list of all feature ranking fields
            this.rankOrdering = undefined;
            // Ordered list of all feature metadata fields
            this.featureMetadataFields = undefined;
            // Ordered, combined list of feature ranking and metadata fields --
            // used in populating the DataTables
            this.featureColumns = undefined;

            // The human-readable "type" of the feature rankings (should be
            // either "Differential" or "Feature Loading")
            this.rankType = undefined;

            this.rankPlotView = undefined;
            this.samplePlotView = undefined;

            // Save the JSONs that will be used to create the visualization.
            this.rankPlotJSON = rankPlotJSON;
            this.samplePlotJSON = samplePlotJSON;
        }

        /* Calls makeRankPlot() and makeSamplePlot(), and waits for them to
         * finish before hiding the loadingMessage.
         *
         * The structure of the async/await usage here is based on the
         * concurrentStart() example on
         * https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/async_function.
         */
        async makePlots() {
            // Note that this will fail if either makePlot function fails with
            // an error. This should be ok for Qurro's purposes, though.
            await Promise.all([this.makeRankPlot(), this.makeSamplePlot()]);

            this.setUpDOM();
            document
                .getElementById("loadingMessage")
                .classList.add("invisible");
        }

        setUpDOM() {
            // All DOM elements that we disable/enable when switching to/from
            // "boxplot mode." We disable these when in "boxplot mode" because
            // Vega-Lite gets grumpy when you try to apply colors to a boxplot
            // when the colors have different granularity than the boxplot's
            // current x-axis. (It does the same thing with tooltips, which is
            // why we delete tooltips also when switching to boxplots.)
            this.colorEles = ["colorField", "colorScale"];

            // Set up relevant DOM bindings
            var display = this;
            // NOTE: ideally we'd update a few of these callbacks to just refer
            // to te original function -- but we need to be able to refer to
            // "this" via the closure including the "display" variable, and I
            // haven't found a good way to do that aside from just declaring
            // individual functions.
            this.elementsWithOnClickBindings = dom_utils.setUpDOMBindings({
                multiFeatureButton: async function() {
                    await display.regenerateFromFiltering();
                },
                autoSelectButton: async function() {
                    await display.regenerateFromAutoSelection();
                },
                exportDataButton: function() {
                    display.exportData();
                }
            });
            this.elementsWithOnChangeBindings = dom_utils.setUpDOMBindings(
                {
                    xAxisField: async function() {
                        await display.updateSamplePlotField("xAxis");
                    },
                    colorField: async function() {
                        await display.updateSamplePlotField("color");
                    },
                    xAxisScale: async function() {
                        await display.updateSamplePlotScale("xAxis");
                    },
                    colorScale: async function() {
                        await display.updateSamplePlotScale("color");
                    },
                    rankField: async function() {
                        await display.updateRankField();
                    },
                    barSizeSlider: async function() {
                        await display.updateRankPlotBarSizeToSlider(true);
                    },
                    fitBarSizeCheckbox: async function() {
                        await display.updateRankPlotBarFitting(true);
                    },
                    boxplotCheckbox: async function() {
                        await display.updateSamplePlotBoxplot();
                    },
                    catColorScheme: async function() {
                        await display.updateSamplePlotColorScheme("category");
                    },
                    quantColorScheme: async function() {
                        await display.updateSamplePlotColorScheme("ramp");
                    },
                    rankPlotColorScheme: async function() {
                        await display.updateRankPlotColorScheme();
                    }
                },
                "onchange"
            );
        }

        makeRankPlot(notFirstTime) {
            if (!notFirstTime) {
                this.rankType = this.rankPlotJSON.datasets.qurro_rank_type;
                // Update the rank field label to say either "Differential" or
                // "Feature Loading". I waffled on whether or not to put this
                // here or in setUpDOM(), but I guess this location makes sense
                document.getElementById(
                    "rankFieldLabel"
                ).textContent = this.rankType;

                this.rankOrdering = this.rankPlotJSON.datasets.qurro_rank_ordering;
                dom_utils.populateSelect(
                    "rankField",
                    this.rankOrdering,
                    this.rankOrdering[0]
                );
                this.featureMetadataFields = this.rankPlotJSON.datasets.qurro_feature_metadata_ordering;
                var searchableFields = {
                    standalone: ["Feature ID"],
                    "Feature Metadata": this.featureMetadataFields
                };
                searchableFields[this.rankType + "s"] = this.rankOrdering;
                dom_utils.populateSelect(
                    "topSearch",
                    searchableFields,
                    "Feature ID",
                    true
                );
                dom_utils.populateSelect(
                    "botSearch",
                    searchableFields,
                    "Feature ID",
                    true
                );
                // Initialize tables and update them
                var columns = [{ title: "Feature ID" }];
                $.each(
                    $.merge(this.rankOrdering, this.featureMetadataFields),
                    function(index, value) {
                        columns.push({ title: value });
                    }
                );
                this.featureColumns = columns;

                // Shared configuration between the numerator and denominator
                // feature DataTables. We define this down here (and not in the
                // RRVDisplay constructor, for example) because we need access
                // to this.featureColumns.
                var dtConfig = {
                    scrollY: "200px",
                    paging: false,
                    scrollX: true,
                    scrollCollapse: true,
                    columns: this.featureColumns,
                    data: [],
                    columnDefs: [{ width: "20%", targets: 0 }],
                    fixedColumns: true
                };
                $("#topFeaturesDisplay").DataTable(dtConfig);
                $("#botFeaturesDisplay").DataTable(dtConfig);
                this.updateFeaturesDisplays(false, true);
                // Figure out which bar size type to default to.
                // We determine this based on how many features there are.
                // This is intended to address cases where there are only a few
                // ranked features (e.g. the matching test) -- in these cases,
                // fitting actually increases the bar sizes to be reasonable to
                // view/select.
                // TODO: make this a separate func so we can unit-test it
                if (
                    this.featureIDs.length <=
                    this.rankPlotJSON.config.view.width
                ) {
                    document.getElementById(
                        "fitBarSizeCheckbox"
                    ).checked = true;
                    this.updateRankPlotBarFitting(false);
                }
            }
            // Set the y-axis to say "Magnitude: [ranking title]" instead of
            // just "[rank title]". Use of "Magnitude" here is based on
            // discussion in issue #191.
            this.rankPlotJSON.encoding.y.title =
                this.rankType + ": " + this.rankPlotJSON.encoding.y.field;
            // We can use a closure to allow callback functions to access "this"
            // (and thereby change the properties of instances of the RRVDisplay
            // class). See https://stackoverflow.com/a/5106369/10730311.
            var parentDisplay = this;
            // We specify a "custom" theme which matches with the
            // "custom"-theme tooltip CSS.
            return vegaEmbed("#rankPlot", this.rankPlotJSON, {
                downloadFileName: "rank_plot",
                tooltip: { theme: "custom" }
            }).then(function(result) {
                parentDisplay.rankPlotView = result.view;
                parentDisplay.addClickEventToRankPlotView(parentDisplay);
            });
        }

        addClickEventToRankPlotView(display) {
            // Set callbacks to let users make selections in the ranks plot
            display.rankPlotView.addEventListener("click", function(e, i) {
                if (i !== null && i !== undefined) {
                    if (i.mark.marktype === "rect") {
                        if (display.onHigh) {
                            display.oldFeatureHigh = display.newFeatureHigh;
                            display.newFeatureHigh = i.datum;
                            console.log(
                                "Set newFeatureHigh: " + display.newFeatureHigh
                            );
                        } else {
                            display.oldFeatureLow = display.newFeatureLow;
                            display.newFeatureLow = i.datum;
                            console.log(
                                "Set newFeatureLow: " + display.newFeatureLow
                            );
                            display.regenerateFromClicking();
                        }
                        display.onHigh = !display.onHigh;
                    }
                }
            });
        }

        /* Calls vegaEmbed() on this.samplePlotJSON.
         *
         * If notFirstTime is falsy, this will initialize some important
         * properties of this RRVDisplay object related to the sample plot
         * (e.g. the metadata columns and feature count information).
         *
         * If you're just calling this function to remake the sample plot with
         * one thing changed (e.g. to change a scale), then it's best to set
         * notFirstTime to true -- in which case this won't do that extra work.
         */
        makeSamplePlot(notFirstTime) {
            if (!notFirstTime) {
                this.metadataCols = this.samplePlotJSON.datasets.qurro_sample_metadata_fields;
                // Note that we set the default metadata fields based on whatever
                // the JSON has as the defaults.
                dom_utils.populateSelect(
                    "xAxisField",
                    this.metadataCols,
                    this.samplePlotJSON.encoding.x.field
                );
                dom_utils.populateSelect(
                    "colorField",
                    this.metadataCols,
                    this.samplePlotJSON.encoding.color.field
                );
            }
            this.updateSamplePlotTooltips();
            this.updateSamplePlotFilters();

            this.updateFieldDroppedSampleStats("x");
            this.updateFieldDroppedSampleStats("color");
            dom_utils.updateMainSampleShownDiv(
                this.droppedSamples,
                this.sampleCount
            );

            var parentDisplay = this;
            return vegaEmbed("#samplePlot", this.samplePlotJSON, {
                downloadFileName: "sample_plot"
            }).then(function(result) {
                parentDisplay.samplePlotView = result.view;
            });
        }

        /* Finds the invalid sample IDs for a given encoding, updates the
         * corresponding <div>, and then updates this.droppedSamples.
         *
         * The input "encoding" should be either "x" or "color". In the future,
         * if other encodings in the sample plot are variable (e.g. size?), it
         * shouldn't be too difficult to update this function to support these.
         */
        updateFieldDroppedSampleStats(encoding) {
            var divID, reason;
            if (encoding === "x") {
                divID = "xAxisSamplesDroppedDiv";
                reason = "xAxis";
            } else if (encoding === "color") {
                divID = "colorSamplesDroppedDiv";
                reason = "color";
            }
            var invalidSampleIDs = this.getInvalidSampleIDs(
                this.samplePlotJSON.encoding[encoding].field,
                encoding
            );
            dom_utils.updateSampleDroppedDiv(
                invalidSampleIDs,
                this.sampleCount,
                divID,
                reason,
                this.samplePlotJSON.encoding[encoding].field
            );
            this.droppedSamples[reason] = invalidSampleIDs;
        }

        // Given a "row" of data about a rank, return its new classification depending
        // on the new selection that just got made.
        updateRankColorSingle(rankRow) {
            if (rankRow["Feature ID"] === this.newFeatureHigh["Feature ID"]) {
                if (
                    rankRow["Feature ID"] === this.newFeatureLow["Feature ID"]
                ) {
                    return "Both";
                } else {
                    return "Numerator";
                }
            } else if (
                rankRow["Feature ID"] === this.newFeatureLow["Feature ID"]
            ) {
                return "Denominator";
            } else {
                return "None";
            }
        }

        updateRankColorMulti(rankRow) {
            var inTop = false;
            var inBot = false;
            for (var i = 0; i < this.topFeatures.length; i++) {
                if (
                    this.topFeatures[i]["Feature ID"] === rankRow["Feature ID"]
                ) {
                    inTop = true;
                    break;
                }
            }
            for (var j = 0; j < this.botFeatures.length; j++) {
                if (
                    this.botFeatures[j]["Feature ID"] === rankRow["Feature ID"]
                ) {
                    inBot = true;
                    break;
                }
            }
            if (inTop) {
                if (inBot) {
                    return "Both";
                } else {
                    return "Numerator";
                }
            } else if (inBot) {
                return "Denominator";
            } else {
                return "None";
            }
        }

        async updateRankField() {
            var newRank = document.getElementById("rankField").value;
            this.rankPlotJSON.encoding.y.field = newRank;
            // NOTE that this assumes that the rank plot only has one transform
            // being used, and that it's a "rank" window transform. (This is a
            // reasonable assumption, since we generate the rank plot.)
            this.rankPlotJSON.transform[0].sort[0].field = newRank;
            await this.remakeRankPlot();
        }

        async remakeRankPlot() {
            this.destroy(true, false, false);
            await this.makeRankPlot(true);
        }

        /* Syncs up the rank plot's bar width with whatever the slider says. */
        async updateRankPlotBarSizeToSlider(callRemakeRankPlot) {
            var sliderBarSize = Number(
                document.getElementById("barSizeSlider").value
            );
            await this.updateRankPlotBarSize(sliderBarSize, callRemakeRankPlot);
        }

        /* Either enables or disables "fitting" the bar widths.
         *
         * Adjusts the "disabled" status of the barSizeSlider accordingly --
         * this prevents users from triggering onchange events while "fitting"
         * the bar widths is enabled.
         */
        async updateRankPlotBarFitting(callRemakeRankPlot) {
            if (document.getElementById("fitBarSizeCheckbox").checked) {
                var fittedBarSize =
                    this.rankPlotJSON.config.view.width /
                    this.featureIDs.length;
                document.getElementById("barSizeSlider").disabled = true;
                await this.updateRankPlotBarSize(
                    fittedBarSize,
                    callRemakeRankPlot
                );
            } else {
                document.getElementById("barSizeSlider").disabled = false;
                await this.updateRankPlotBarSizeToSlider(callRemakeRankPlot);
            }
        }

        /* Remakes the rank plot with the specified bar width (in pixels).
         *
         * If newBarSize < 1, this also makes the barSizeWarning element
         * visible. (If newBarSize >= 1, this will make the barSizeWarning
         * element invisible.)
         */
        async updateRankPlotBarSize(newBarSize, callRemakeRankPlot) {
            this.rankPlotJSON.encoding.x.scale.rangeStep = newBarSize;
            if (newBarSize < 1) {
                document
                    .getElementById("barSizeWarning")
                    .classList.remove("invisible");
            } else {
                document
                    .getElementById("barSizeWarning")
                    .classList.add("invisible");
            }
            if (callRemakeRankPlot) {
                await this.remakeRankPlot();
            }
        }

        /* Updates the selected log-ratio, which involves updating both plots:
         *
         * 1) update sample log-ratios in the sample plot
         * 2) update the "classifications" of features in the rank plot
         * 3) update dropped sample information re: the new log-ratios
         * */
        async updateLogRatio(updateBalanceFunc, updateRankColorFunc) {
            var dataName = this.samplePlotJSON.data.name;
            var parentDisplay = this;
            var nullBalanceSampleIDs = [];

            var samplePlotViewChanged = this.samplePlotView.change(
                dataName,
                vega.changeset().modify(
                    /* Calculate the new balance for each sample.
                     *
                     * For reference, the use of modify() here is based on
                     * https://github.com/vega/vega/issues/1028#issuecomment-334295328
                     * (This is where I learned that
                     * vega.changeset().modify() existed.)
                     * Also, vega.truthy is a utility function: it just
                     * returns true.
                     */
                    vega.truthy,
                    "qurro_balance",
                    // function to run to determine what the new balances are
                    function(sampleRow) {
                        var sampleBalance = updateBalanceFunc.call(
                            parentDisplay,
                            sampleRow
                        );
                        if (sampleBalance === null) {
                            nullBalanceSampleIDs.push(sampleRow["Sample ID"]);
                        }
                        return sampleBalance;
                    }
                )
            );

            // Update rank plot based on the new log-ratio
            // Doing this alongside the change to the sample plot is done so that
            // the "states" of the plot re: selected features + sample log
            // ratios are unified.
            var rankDataName = this.rankPlotJSON.data.name;
            // While we're doing this, keep track of how many features have a
            // log-ratio classification of "Both" (i.e. they're in both the
            // numerator and denominator). Since this is Likely A Problem (TM),
            // we want to warn the user about these features.
            var bothFeatureCount = 0;
            var rankPlotViewChanged = this.rankPlotView.change(
                rankDataName,
                vega
                    .changeset()
                    .modify(vega.truthy, "qurro_classification", function(
                        rankRow
                    ) {
                        var color = updateRankColorFunc.call(
                            parentDisplay,
                            rankRow
                        );
                        if (color === "Both") {
                            bothFeatureCount++;
                        }
                        return color;
                    })
            );

            // Change both the plots, and move on when these changes are done.
            await Promise.all([
                samplePlotViewChanged.runAsync(),
                rankPlotViewChanged.runAsync()
            ]);

            // Now that the plots have been updated, update the dropped sample
            // count re: the new sample log-ratios.
            this.droppedSamples.balance = nullBalanceSampleIDs;
            dom_utils.updateMainSampleShownDiv(
                this.droppedSamples,
                this.sampleCount
            );

            dom_utils.updateSampleDroppedDiv(
                nullBalanceSampleIDs,
                this.sampleCount,
                "balanceSamplesDroppedDiv",
                "balance"
            );

            // And hide / show the "both" warning as needed.
            if (bothFeatureCount > 0) {
                // Yeah, yeah, I know setting .innerHTML using variables is bad
                // practice, because if the variable(s) in question have weird
                // characters then this can result in code injection/etc.
                // However, bothFeatureCount should always be a number, so this
                // shouldn't be a problem here.
                document.getElementById("commonFeatureWarning").innerHTML =
                    "<strong>Warning:</strong> Currently, " +
                    vega.stringValue(bothFeatureCount) +
                    " feature(s) " +
                    "are selected in <strong>both</strong> the numerator " +
                    "and denominator of the log-ratio. We strongly suggest " +
                    "you instead look at a log-ratio that doesn't contain " +
                    "common features in the numerator and denominator.";
                document
                    .getElementById("commonFeatureWarning")
                    .classList.remove("invisible");
            } else {
                document
                    .getElementById("commonFeatureWarning")
                    .classList.add("invisible");
            }
        }

        /* Updates the rank and sample plot based on "autoselection."
         *
         * By "autoselection," we just mean picking the top/bottom features for
         * the current ranking in the rank plot.
         */
        async regenerateFromAutoSelection() {
            var inputNumber = document.getElementById("autoSelectNumber").value;
            // autoSelectType should be either "autoPercent" or "autoLiteral".
            // Therefore, there are four possible values of this we can pass in
            // to filterFeatures:
            // -autoPercentTop
            // -autoPercentBot
            // -autoLiteralTop
            // -autoLiteralBot
            var autoSelectType = document.getElementById("autoSelectType")
                .value;
            this.topFeatures = feature_computation.filterFeatures(
                this.rankPlotJSON,
                inputNumber,
                this.rankPlotJSON.encoding.y.field,
                autoSelectType + "Top"
            );
            this.botFeatures = feature_computation.filterFeatures(
                this.rankPlotJSON,
                inputNumber,
                this.rankPlotJSON.encoding.y.field,
                autoSelectType + "Bot"
            );
            // TODO: abstract below stuff to a helper function for use by
            // regenerateFromAutoSelection() and RegenerateFromFiltering()
            this.updateFeaturesDisplays();
            await this.updateLogRatio(
                this.updateBalanceMulti,
                this.updateRankColorMulti
            );
        }

        /* Updates the rank and sample plot based on the "filtering" controls.
         *
         * Broadly, this just involves applying user-specified queries to get a
         * list of feature(s) for the numerator and denominator of a log-ratio.
         *
         * This then calls updateLogRatio().
         */
        async regenerateFromFiltering() {
            // Determine which feature field(s) (Feature ID, anything in the
            // feature metadata, anything in the feature rankings) to look at
            var topField = document.getElementById("topSearch").value;
            var botField = document.getElementById("botSearch").value;
            var topSearchType = document.getElementById("topSearchType").value;
            var botSearchType = document.getElementById("botSearchType").value;
            var topEnteredText = document.getElementById("topText").value;
            var botEnteredText = document.getElementById("botText").value;
            this.topFeatures = feature_computation.filterFeatures(
                this.rankPlotJSON,
                topEnteredText,
                topField,
                topSearchType
            );
            this.botFeatures = feature_computation.filterFeatures(
                this.rankPlotJSON,
                botEnteredText,
                botField,
                botSearchType
            );
            this.updateFeaturesDisplays();
            await this.updateLogRatio(
                this.updateBalanceMulti,
                this.updateRankColorMulti
            );
        }

        async regenerateFromClicking() {
            if (
                this.newFeatureLow !== undefined &&
                this.newFeatureHigh !== undefined
            ) {
                if (
                    this.newFeatureLow !== null &&
                    this.newFeatureHigh !== null
                ) {
                    // We wrap this stuff in checks because it's conceivable
                    // that we've reached this point and oldFeatureLow /
                    // oldFeatureHigh are still undefined/null. However, we
                    // expect that at least the new features should be actual
                    // feature row objects (i.e. with "Feature ID" properties).
                    var lowsDiffer = true;
                    var highsDiffer = true;
                    if (
                        this.oldFeatureLow !== null &&
                        this.oldFeatureLow !== undefined
                    ) {
                        lowsDiffer =
                            this.oldFeatureLow["Feature ID"] !=
                            this.newFeatureLow["Feature ID"];
                    }
                    if (
                        this.oldFeatureHigh !== null &&
                        this.oldFeatureHigh !== undefined
                    ) {
                        highsDiffer =
                            this.oldFeatureHigh["Feature ID"] !=
                            this.newFeatureHigh["Feature ID"];
                    }
                    if (lowsDiffer || highsDiffer) {
                        this.updateFeaturesDisplays(true);
                        // Time to update the plots re: the new log-ratio
                        await this.updateLogRatio(
                            this.updateBalanceSingle,
                            this.updateRankColorSingle
                        );
                    }
                }
            }
        }

        updateFeatureHeaderCounts(topCt, botCt) {
            var featureCt = this.featureIDs.length;
            var featureCtStr = featureCt.toLocaleString();
            document.getElementById("numHeader").textContent =
                "Numerator Features: " +
                topCt.toLocaleString() +
                " / " +
                featureCtStr +
                " (" +
                dom_utils.formatPercentage(topCt, featureCt) +
                "%) selected";
            document.getElementById("denHeader").textContent =
                "Denominator Features: " +
                botCt.toLocaleString() +
                " / " +
                featureCtStr +
                " (" +
                dom_utils.formatPercentage(botCt, featureCt) +
                "%) selected";
        }

        /* Updates the DataTables (formerly textareas, in versions of Qurro
         * before 0.5.0) that list the selected features, as well as
         * the corresponding header elements that indicate the numbers of
         * selected features.
         *
         * This defaults to updating based on the "multiple" selections'
         * values. If you pass in a truthy value for the clear argument,
         * this will instead clear these text areas; if you pass in a truthy
         * value for the single argument (and clear is falsy), this will
         * instead update based on the single selection values.
         */
        updateFeaturesDisplays(single, clear) {
            var topDisplay = $("#topFeaturesDisplay").DataTable();
            var botDisplay = $("#botFeaturesDisplay").DataTable();

            topDisplay.clear().draw();
            botDisplay.clear().draw();

            if (clear) {
                this.updateFeatureHeaderCounts(0, 0);
            } else {
                var topFeatureList, botFeatureList;
                if (single) {
                    topFeatureList = [this.newFeatureHigh];
                    botFeatureList = [this.newFeatureLow];
                } else {
                    topFeatureList = this.topFeatures;
                    botFeatureList = this.botFeatures;
                }
                this.updateFeatureHeaderCounts(
                    topFeatureList.length,
                    botFeatureList.length
                );

                // Keep track of feature columns via a closure so that we can
                // reference it from inside the following function(...s)
                var columns = this.featureColumns;
                $.each(topFeatureList, function(index, feature) {
                    topDisplay.row.add(
                        RRVDisplay.getRowOfColumnData(feature, columns)
                    );
                });
                $.each(botFeatureList, function(index, feature) {
                    botDisplay.row.add(
                        RRVDisplay.getRowOfColumnData(feature, columns)
                    );
                });

                topDisplay.draw();
                botDisplay.draw();
            }
        }

        /* Converts a "feature row" (from a V-L spec) to a DataTables-ok row.
         *
         * I moved this to a separate function so that jshint would stop
         * yelling at me for declaring a function inside a block ._.
         */
        static getRowOfColumnData(feature, columns) {
            var row = [];
            $.each(columns, function(index, column) {
                row.push(feature[column.title]);
            });
            return row;
        }

        updateSamplePlotTooltips() {
            // NOTE: this should be safe from duplicate entries within
            // tooltips so long as you don't change the field titles
            // displayed.
            this.samplePlotJSON.encoding.tooltip = [
                { type: "nominal", field: "Sample ID" },
                {
                    type: "quantitative",
                    field: "qurro_balance",
                    title: "Current Natural Log-Ratio"
                },
                {
                    type: this.samplePlotJSON.encoding.x.type,
                    field: this.samplePlotJSON.encoding.x.field
                },
                {
                    type: this.samplePlotJSON.encoding.color.type,
                    field: this.samplePlotJSON.encoding.color.field
                }
            ];
        }

        /* Modifies the transform property of the sample plot JSON to include a
         * filter based on the currently-used x-axis and color fields.
         *
         * This results of this filter should corroborate the result of
         * getInvalidSampleIDs(). From testing, it looks like the only other
         * values that would automatically be filtered out would be NaN /
         * undefined values, and none of those should show up in the sample
         * metadata values due to how Qurro's python code processes sample
         * metadata (replacing all NaN values with None, which gets converted
         * to null by json.dumps() in python).
         */
        updateSamplePlotFilters() {
            // Figure out the current [x-axis/color] [field/encoding type].
            // Note that we explicitly wrap the fields in double-quotes, so
            // even field names with weird characters shouldn't be able to mess
            // this up.
            var datumXField =
                "datum[" +
                vega.stringValue(this.samplePlotJSON.encoding.x.field) +
                "]";
            var datumColorField =
                "datum[" +
                vega.stringValue(this.samplePlotJSON.encoding.color.field) +
                "]";
            var xType = this.samplePlotJSON.encoding.x.type;
            var colorType = this.samplePlotJSON.encoding.color.type;

            var filterString = "datum.qurro_balance != null";
            // NOTE: if the current x and color fields are the same, there will
            // be some redundancy in filterString. Might be worth addressing
            // this in the future, but shouldn't be a big deal -- Vega* doesn't
            // seem to mind.
            filterString += " && " + datumXField + " != null";
            filterString += " && " + datumColorField + " != null";

            if (xType === "quantitative") {
                filterString += " && isFinite(toNumber(" + datumXField + "))";
            }
            if (colorType === "quantitative") {
                filterString +=
                    " && isFinite(toNumber(" + datumColorField + "))";
            }

            this.samplePlotJSON.transform = [{ filter: filterString }];
        }

        /* Update color so that color encoding matches the x-axis encoding
         * (due to how box plots work in Vega-Lite). To be clear, we also
         * update the color field <select> to show the user what's going on.
         */
        setColorForBoxplot() {
            var category = this.samplePlotJSON.encoding.x.field;
            this.samplePlotJSON.encoding.color.field = category;
            document.getElementById("colorField").value = category;
            document.getElementById("colorScale").value = "nominal";
            this.samplePlotJSON.encoding.color.type = "nominal";
        }

        async updateSamplePlotField(vizAttribute) {
            if (vizAttribute === "xAxis") {
                this.samplePlotJSON.encoding.x.field = document.getElementById(
                    "xAxisField"
                ).value;
                if (
                    document.getElementById("boxplotCheckbox").checked &&
                    this.samplePlotJSON.encoding.x.type === "nominal"
                ) {
                    this.setColorForBoxplot();
                }
            } else {
                this.samplePlotJSON.encoding.color.field = document.getElementById(
                    "colorField"
                ).value;
            }
            await this.remakeSamplePlot();
        }

        async remakeSamplePlot() {
            // Clear out the sample plot. NOTE that I'm not sure if this is
            // 100% necessary, but it's probs a good idea to prevent memory
            // waste.
            this.destroy(false, true, false);
            await this.makeSamplePlot(true);
        }

        /* Iterates through every sample in the sample plot JSON and
         * looks at the sample's fieldName field. Returns a list of "invalid"
         * sample IDs -- i.e. those that, based on the current field and
         * corresponding encoding (e.g. "color" or "x"), can't be displayed in
         * the sample plot even if their other properties (balance, other
         * encodings) are valid.
         *
         * The "validity" of a sample is computed via the following checks:
         * 1. The sample's fieldName field must not be null
         *    (we don't bother explicitly checking for NaN, "", strings
         *    containing only whitespace, and undefined since they should never
         *    be included in the sample metadata JSON produced by Qurro's python
         *    code)
         * 2. If the corresponding encoding type is quantitative, the sample's
         *    fieldName field must be a finite number (as determined by
         *    isFinite(vega.toNumber(f)), where f is the field value). This
         *    accounts for Infinities/NaNs in the data, which shouldn't appear
         *    literally in the dataset but could potentially sneak in as
         *    strings (e.g. "Infinity", "-Infinity", "NaN") -- we'd display
         *    these strings normally for a nominal encoding, but for a
         *    quantitative encoding we filter them out.
         *
         *    Note that the normal isFinite() (as opposed to Number.isFinite())
         *    has a few quirks, including isFinite(null) and isFinite("   ")
         *    both being true. However, we should avoid these, since we already
         *    check for null values before calling isFinite(), and since the
         *    metadata handlers filter out leading/trailing whitespace (so
         *    inputs like "" or "    " will end up as null in the plot JSONs),
         *    we should get around these quirks. (See the sample stats test for
         *    examples of how Qurro's input handling is good in this way.)
         */
        getInvalidSampleIDs(fieldName, correspondingEncoding) {
            var dataName = this.samplePlotJSON.data.name;
            var currFieldVal;
            var currSampleID;
            var invalidSampleIDs = [];
            for (
                var i = 0;
                i < this.samplePlotJSON.datasets[dataName].length;
                i++
            ) {
                currFieldVal = this.samplePlotJSON.datasets[dataName][i][
                    fieldName
                ];
                currSampleID = this.samplePlotJSON.datasets[dataName][i][
                    "Sample ID"
                ];
                if (currFieldVal !== null) {
                    if (
                        this.samplePlotJSON.encoding[correspondingEncoding]
                            .type === "quantitative"
                    ) {
                        if (!isFinite(vega.toNumber(currFieldVal))) {
                            // scale is quantitative and this isn't a valid
                            // numerical value
                            invalidSampleIDs.push(currSampleID);
                        }
                        // If the above check passed (i.e. this value
                        // is "numerical"), then we'll just continue on in
                        // the loop without adding it to invalidSampleIDs.
                    }
                    // We don't include an "else" branch here, because this
                    // part is only reached if the current encoding is nominal
                    // (in which case we know this sample ID is valid, so we
                    // don't do anything with it).
                    // NOTE: this *assumes* that the scale is either
                    // quantitative or nominal. If it's something like temporal
                    // then this will get messed up, and we'll need to do
                    // something else to address it.
                } else {
                    // currFieldVal *is* null.
                    invalidSampleIDs.push(currSampleID);
                }
            }
            return invalidSampleIDs;
        }

        async updateSamplePlotColorScheme(scaleRangeType) {
            var newScheme;
            var changesCurrentPlot = false;
            if (scaleRangeType === "category") {
                newScheme = document.getElementById("catColorScheme").value;
                changesCurrentPlot =
                    this.samplePlotJSON.encoding.color.type === "nominal";
            } else if (scaleRangeType === "ramp") {
                newScheme = document.getElementById("quantColorScheme").value;
                changesCurrentPlot =
                    this.samplePlotJSON.encoding.color.type === "quantitative";
            } else {
                throw new Error(
                    "Unrecognized scale range type specified: " + scaleRangeType
                );
            }
            this.samplePlotJSON.config.range[scaleRangeType].scheme = newScheme;
            // Only remake the sample plot if the new color scheme would effect
            // the currently displayed colors in the sample plot.
            if (changesCurrentPlot) {
                await this.remakeSamplePlot();
            }
        }

        async updateRankPlotColorScheme() {
            var newColorScheme = document
                .getElementById("rankPlotColorScheme")
                .value.split(",");
            this.rankPlotJSON.encoding.color.scale.range[1] = newColorScheme[0];
            this.rankPlotJSON.encoding.color.scale.range[2] = newColorScheme[1];
            this.rankPlotJSON.encoding.color.scale.range[3] = newColorScheme[2];
            await this.remakeRankPlot();
        }

        /* Changes the scale type of either the x-axis or colorization in the
         * sample plot. This isn't doable with Vega signals -- we need to
         * literally reload the Vega-Lite specification with the new scale
         * type in order to make these changes take effect.
         */
        async updateSamplePlotScale(vizAttribute) {
            if (vizAttribute === "xAxis") {
                var newScale = document.getElementById("xAxisScale").value;
                this.samplePlotJSON.encoding.x.type = newScale;
                // This assumes that the x-axis specification only has the
                // labelAngle parameter.
                if (newScale === "nominal") {
                    this.samplePlotJSON.encoding.x.axis = { labelAngle: -45 };
                    if (document.getElementById("boxplotCheckbox").checked) {
                        this.changeSamplePlotToBoxplot(false);
                    }
                } else {
                    this.changeSamplePlotFromBoxplot(false);
                    // This should work even if the axis property is undefined
                    // -- it just won't do anything in that case.
                    delete this.samplePlotJSON.encoding.x.axis;
                }
            } else {
                this.samplePlotJSON.encoding.color.type = document.getElementById(
                    "colorScale"
                ).value;
            }
            await this.remakeSamplePlot();
        }

        async updateSamplePlotBoxplot() {
            // We only bother changing up anything if the sample plot x-axis
            // is currently categorical.
            if (this.samplePlotJSON.encoding.x.type === "nominal") {
                if (document.getElementById("boxplotCheckbox").checked) {
                    await this.changeSamplePlotToBoxplot(true);
                } else {
                    await this.changeSamplePlotFromBoxplot(true);
                }
            }
        }

        /* Changes the sample plot JSON and DOM elements to get ready for
         * switching to "boxplot mode." If callRemakeSamplePlot is truthy, this
         * will actually call this.remakeSamplePlot(); otherwise, this won't do
         * anything.
         *
         * callRemakeSamplePlot should be false if this is called in the
         * middle of remaking the sample plot, anyway -- e.g. if the user
         * switched the x-axis scale type from quantitative to categorical, and
         * the "use boxplots" checkbox was already checked.
         *
         * callRemakeSamplePlot should be true if this is called as the only
         * update to the sample plot that's going to be made -- i.e. the user
         * was already using a categorical x-axis scale, and they just clicked
         * the "use boxplots" checkbox.
         */
        async changeSamplePlotToBoxplot(callRemakeSamplePlot) {
            this.samplePlotJSON.mark.type = "boxplot";
            // Make the middle tick of the boxplot black. This makes boxes for
            // which only one sample is available show up on the white
            // background and light-gray axis.
            this.samplePlotJSON.mark.median = { color: "#000000" };
            dom_utils.changeElementsEnabled(this.colorEles, false);
            this.setColorForBoxplot();
            if (callRemakeSamplePlot) {
                await this.remakeSamplePlot();
            }
        }

        /* Like changeSamplePlotToBoxplot(), but the other way around. This is
         * a bit simpler, since (as of writing) we have to do less to go back
         * to a normal circle mark from the boxplot mark.
         *
         * callRemakeSamplePlot works the same way as in
         * changeSamplePlotToBoxplot().
         */
        async changeSamplePlotFromBoxplot(callRemakeSamplePlot) {
            this.samplePlotJSON.mark.type = "circle";
            delete this.samplePlotJSON.mark.median;
            dom_utils.changeElementsEnabled(this.colorEles, true);
            // No need to explicitly adjust color or tooltips here; tooltips
            // will be auto-added in updateSamplePlotTooltips(), and color
            // should have been kept up-to-date every time the field was
            // changed while boxplot mode was going on (as well as at the
            // start of boxplot mode), in setColorForBoxplot().
            if (callRemakeSamplePlot) {
                await this.remakeSamplePlot();
            }
        }

        static identifySampleIDs(samplePlotSpec) {
            var sampleIDs = [];
            var dataName = samplePlotSpec.data.name;
            var sid;
            for (var s = 0; s < samplePlotSpec.datasets[dataName].length; s++) {
                sid = samplePlotSpec.datasets[dataName][s]["Sample ID"];
                if (sid !== undefined) {
                    sampleIDs.push(sid);
                }
            }
            return sampleIDs;
        }

        /* Checks if a sample ID is actually supported by the count data we
         * have. We do this by just looking at all the samples with count data
         * for a feature ID, and checking to make sure that the sample ID is
         * one of those.
         *
         * (This function makes the assumption that each feature will have the
         * same number of samples associated with it -- this is why we only
         * bother checking a single feature here. This is a safe assumption,
         * since we construct the feature count JSON from a BIOM table on the
         * python side of things.)
         */
        validateSampleID(sampleID) {
            if (!this.sampleIDs.includes(sampleID)) {
                throw new Error("Invalid sample ID: " + sampleID);
            }
        }

        /* Gets count data from the featureCts object. This uses a sparse
         * storage method, so only samples with a nonzero count for a given
         * feature are contained in that feature's entry.
         *
         * So, if the "entry" for a sample for a feature in featureCts is falsy
         * (i.e. undefined, but could also be false, 0, "", null, or NaN --
         * none of these should ever occur in the count JSON but if they do
         * that should also be indicative of a zero count [1]), then we
         * consider that sample's count to be 0. Otherwise, we just return the
         * entry.
         *
         * [1] See https://developer.mozilla.org/en-US/docs/Glossary/Truthy
         */
        getCount(featureID, sampleID) {
            var putativeCount = this.featureCts[featureID][sampleID];
            if (putativeCount) {
                return putativeCount;
            } else {
                return 0;
            }
        }

        /* Given a "row" of the sample plot's JSON for a sample, and given an array of
         * features, return the sum of the sample's abundances for those particular features.
         * TODO: add option to do log geometric means
         */
        sumAbundancesForSampleFeatures(sampleRow, features) {
            var sampleID = sampleRow["Sample ID"];
            this.validateSampleID(sampleID);
            var abundance = 0;
            for (var t = 0; t < features.length; t++) {
                abundance += this.getCount(features[t]["Feature ID"], sampleID);
            }
            return abundance;
        }

        /* Use abundance data to compute the new log-ratio ("balance") values of
         * log(high feature abundance) - log(low feature abundance) for a given sample.
         *
         * This particular function is for log-ratios of two individual features that were
         * selected via the rank plot.
         */
        updateBalanceSingle(sampleRow) {
            var sampleID = sampleRow["Sample ID"];
            this.validateSampleID(sampleID);
            var topCt = this.getCount(
                this.newFeatureHigh["Feature ID"],
                sampleID
            );
            var botCt = this.getCount(
                this.newFeatureLow["Feature ID"],
                sampleID
            );
            return feature_computation.computeBalance(topCt, botCt);
        }

        /* Like updateBalanceSingle, but considers potentially many features in the
         * numerator and denominator of the log-ratio. For log-ratios generated
         * by textual queries.
         */
        updateBalanceMulti(sampleRow) {
            this.validateSampleID(sampleRow["Sample ID"]);
            // NOTE: For multiple features Virus/Staphylococcus:
            // test cases in comparison to first scatterplot in Jupyter
            // Notebook: 1517, 1302.
            var topCt = this.sumAbundancesForSampleFeatures(
                sampleRow,
                this.topFeatures
            );
            var botCt = this.sumAbundancesForSampleFeatures(
                sampleRow,
                this.botFeatures
            );
            return feature_computation.computeBalance(topCt, botCt);
        }

        /* Calls dom_utils.downloadDataURI() on the result of
         * getSamplePlotData().
         */
        exportData() {
            var tsv = this.getSamplePlotData(
                this.samplePlotJSON.encoding.x.field,
                this.samplePlotJSON.encoding.color.field
            );
            dom_utils.downloadDataURI("sample_plot_data.tsv", tsv, true);
            // Also I guess export feature IDs somehow.
            // TODO go through this.topFeatures and this.botFeatures; convert
            // from two arrays to a string, where each feature is separated by
            // a newline and the numerator feature list is followed by
            // "DENOMINATOR FEATURES\n" and then the denominator feature list.
            // Then I guess uh just save that to a .txt file?
        }

        /* Adds surrounding quotes if the string t contains any whitespace or
         * contains any double-quote characters (").
         *
         * If surrounding quotes are added, this will also "escape" any double
         * quote characters in t by converting each double quote to 2 double
         * quotes. e.g. abcd"ef"g --> "abcd""ef""g"
         *
         * This should make t adhere to the excel-tab dialect of python's csv
         * module, as discussed in the QIIME 2 documentation
         * (https://docs.qiime2.org/2019.1/tutorials/metadata/#tsv-dialect-and-parser)
         * and elaborated on in PEP 305
         * (https://www.python.org/dev/peps/pep-0305/).
         */
        static quoteTSVFieldIfNeeded(t) {
            // Use of regex .test() with \s per
            // https://stackoverflow.com/a/1731200/10730311
            if (typeof t === "string" && /\s|"/g.test(t)) {
                // If the first argument of .replace() is just a string, only
                // the first match will be changed. Using a regex with the g
                // flag fixes this; see
                // https://stackoverflow.com/a/10610408/10730311
                return '"' + t.replace(/"/g, '""') + '"';
            } else {
                return t;
            }
        }

        /* Exports data from the sample plot to a string that can be written to
         * a .tsv file for further analysis of these data.
         *
         * If no points have been "drawn" on the sample plot -- i.e. they all
         * have a qurro_balance attribute of null -- then this just returns an
         * empty string.
         */
        getSamplePlotData(currXField, currColorField) {
            var outputTSV =
                '"Sample ID"\tCurrent_Natural_Log_Ratio\t' +
                RRVDisplay.quoteTSVFieldIfNeeded(currXField) +
                "\t" +
                RRVDisplay.quoteTSVFieldIfNeeded(currColorField);
            var dataName = this.samplePlotJSON.data.name;
            // Get all of the data available to the sample plot
            // (Note that updateLogRatio() causes updates to samples'
            // qurro_balance properties, so we don't have to use the
            // samplePlotView)
            var data = this.samplePlotJSON.datasets[dataName];
            var currSampleID, currXValue, currColorValue;
            for (var i = 0; i < data.length; i++) {
                currSampleID = RRVDisplay.quoteTSVFieldIfNeeded(
                    data[i]["Sample ID"]
                );
                outputTSV +=
                    "\n" + currSampleID + "\t" + String(data[i].qurro_balance);
                currXValue = RRVDisplay.quoteTSVFieldIfNeeded(
                    String(data[i][currXField])
                );
                currColorValue = RRVDisplay.quoteTSVFieldIfNeeded(
                    String(data[i][currColorField])
                );
                outputTSV += "\t" + currXValue + "\t" + currColorValue;
            }
            return outputTSV;
        }

        /* Selectively clears the effects of this rrv instance on the DOM.
         *
         * You should only call this with clearOtherStuff set to a truthy value
         * when you want to get rid of the entire current display. This won't
         * really "delete" the current RRVDisplay instance, but this should
         * make it feasible to create new RRVDisplay instances afterwards
         * without refreshing the page.
         */
        destroy(clearRankPlot, clearSamplePlot, clearOtherStuff) {
            if (clearRankPlot) {
                this.rankPlotView.finalize();
                dom_utils.clearDiv("rankPlot");
            }
            if (clearSamplePlot) {
                this.samplePlotView.finalize();
                dom_utils.clearDiv("samplePlot");
            }
            if (clearOtherStuff) {
                // Clear the bindings of bound DOM elements
                for (
                    var i = 0;
                    i < this.elementsWithOnClickBindings.length;
                    i++
                ) {
                    // Based on https://stackoverflow.com/a/53357610/10730311.
                    // Setting .onclick = undefined just straight-up doesn't
                    // work for some reason (even after you do that, the
                    // onclick property is null instead of undefined).
                    // So setting to null is needed in order for a testable way
                    // to "unset" the .onclick property.
                    document.getElementById(
                        this.elementsWithOnClickBindings[i]
                    ).onclick = null;
                }
                for (
                    var j = 0;
                    j < this.elementsWithOnChangeBindings.length;
                    j++
                ) {
                    document.getElementById(
                        this.elementsWithOnChangeBindings[j]
                    ).onchange = null;
                }
                // Reset various UI elements to their "default" states

                // Completely destroy the "features text" displays -- this'll
                // let us re-initialize the DataTable in makeRankPlot() without
                // causing this sort of error:
                // https://datatables.net/manual/tech-notes/3
                $("#topFeaturesDisplay")
                    .DataTable()
                    .destroy();
                $("#botFeaturesDisplay")
                    .DataTable()
                    .destroy();
                dom_utils.clearDiv("topFeaturesDisplay");
                dom_utils.clearDiv("botFeaturesDisplay");

                // Hide (if not already hidden) the warning about feature(s)
                // being in both the numerator and denominator of a log-ratio
                document
                    .getElementById("commonFeatureWarning")
                    .classList.add("invisible");

                // Clear <select>s populated with field information from this
                // RRVDisplay's JSONs
                dom_utils.clearDiv("rankField");
                dom_utils.clearDiv("topSearch");
                dom_utils.clearDiv("botSearch");
                dom_utils.clearDiv("xAxisField");
                dom_utils.clearDiv("colorField");

                // Un-check the boxplot checkbox
                document.getElementById("boxplotCheckbox").checked = false;

                // Set search types to "text"
                document.getElementById("topSearchType").value = "text";
                document.getElementById("botSearchType").value = "text";

                // Clear search input fields
                document.getElementById("topText").value = "";
                document.getElementById("botText").value = "";

                // Set scale type <select>s to default values
                document.getElementById("xAxisScale").value = "nominal";
                document.getElementById("colorScale").value = "nominal";

                // Set color <select>s to default values
                document.getElementById("catColorScheme").value = "tableau10";
                document.getElementById("quantColorScheme").value = "blues";

                // Set bar width controls/elements to default values
                document.getElementById("barSizeSlider").value = "1";
                document.getElementById("barSizeSlider").disabled = false;
                document.getElementById("fitBarSizeCheckbox").checked = false;
                document
                    .getElementById("barSizeWarning")
                    .classList.add("invisible");

                // Clear sample dropped stats divs and set them invisible
                for (var s = 0; s < dom_utils.statDivs.length; s++) {
                    dom_utils.clearDiv(dom_utils.statDivs[s]);
                    document
                        .getElementById(dom_utils.statDivs[s])
                        .classList.add("invisible");
                }

                // Reset the UI elements that have been updated with the
                // rankType. At present, we can do this just by clearing the
                // rankFieldLabel; the only other places the rankType is used
                // are in the rank plot y-axis (which is cleared in destroy()
                // if clearRankPlot is truthy) and in the searching <select>s
                // (which were already cleared via calls to clearDiv()).
                document.getElementById("rankFieldLabel").textContent = "";
            }
        }
    }

    return { RRVDisplay: RRVDisplay };
});
