/* This file contains some methods for manipulating DOM elements in a
 * client-side web interface.
 */
define(["vega"], function(vega) {
    /* Assigns DOM bindings to elements.
     *
     * If eventHandler is set to "onchange", this will update the onchange
     * event handler for these elements. Otherwise, this will update the
     * onclick event handler.
     */
    function setUpDOMBindings(elementID2function, eventHandler) {
        var elementIDs = Object.keys(elementID2function);
        var currID;
        for (var i = 0; i < elementIDs.length; i++) {
            currID = elementIDs[i];
            if (eventHandler === "onchange") {
                document.getElementById(currID).onchange =
                    elementID2function[currID];
            } else {
                document.getElementById(currID).onclick =
                    elementID2function[currID];
            }
        }
        return elementIDs;
    }

    /* Adds <option> elements to a parent DOM element.
     *
     * For each item "x" in optionList, a new <option> element is created
     * inside parentElement with a value and name of "x".
     */
    function addOptionsToParentElement(optionList, parentElement) {
        var optionEle;
        for (var m = 0; m < optionList.length; m++) {
            optionEle = document.createElement("option");
            optionEle.value = optionEle.text = optionList[m];
            parentElement.appendChild(optionEle);
        }
    }

    /* Populates a <select> DOM element with a list or object of options.
     *
     * This will remove any options already present in the <select> first.
     *
     * By default -- if optgroupMap is falsy -- this function assumes that
     * "options" is just a list of options to add directly to the <select>.
     *
     * If optgroupMap is truthy, this assumes that "options" is actually an
     * object of the form {"g1": ["o1", ...], ..., "gX": ["oX", ...]}.
     * This will still populate the <select> with all of the options ("o1",
     * "oX", ...) in these lists, so the behavior will functionally be the
     * same -- but the keys of the "options" object ("g1", "gX", ...) will be
     * used to create <optgroup>s surrounding their respective options. (So
     * "o1" would be an <option> within an <optgroup> labelled "g1" and "oX"
     * would be an <option> within an <optgroup> labelled "gX", for example.)
     *
     * (If one of the <optgroup> names is "standalone", then its children will
     * be added to the <select> directly. This functionality can be used to
     * create "global" options that aren't within a specific <optgroup>.)
     */
    function populateSelect(selectID, options, defaultVal, optgroupMap) {
        var optgroups;
        if (optgroupMap) {
            optgroups = Object.keys(options);
            if (optgroups.length <= 0) {
                throw new Error(
                    "options must have at least one optgroup specified"
                );
            }
            var atLeastOneOption = false;
            for (var i = 0; i < optgroups.length; i++) {
                if (options[optgroups[i]].length > 0) {
                    atLeastOneOption = true;
                    break;
                }
            }
            if (!atLeastOneOption) {
                throw new Error("options must have at least one child option");
            }
        } else {
            if (options.length <= 0) {
                throw new Error("options must have at least one value");
            }
        }
        var optionEle, groupEle;
        var selectEle = document.getElementById(selectID);
        // Remove any options already present in the <select>
        clearDiv(selectID);
        // Actually populate the <select>
        if (optgroupMap) {
            for (var g = 0; g < optgroups.length; g++) {
                // Ignore empty optgroups. (In practice, this means that
                // datasets without any specified feature metadata won't have
                // an empty "Feature Metadata" optgroup shown in the search
                // field <select>s.)
                if (options[optgroups[g]].length > 0) {
                    if (optgroups[g] === "standalone") {
                        // If we find an optgroups with the label "standalone"
                        // then we'll just add the option(s) within that label
                        // to the <select> directly.
                        addOptionsToParentElement(
                            options.standalone,
                            selectEle
                        );
                    } else {
                        // For all other optgroups, though, actually create an
                        // <optgroup> element, populate that, then add that to
                        // the <select>.
                        groupEle = document.createElement("optgroup");
                        groupEle.label = optgroups[g];
                        addOptionsToParentElement(
                            options[optgroups[g]],
                            groupEle
                        );
                        selectEle.appendChild(groupEle);
                    }
                }
            }
        } else {
            addOptionsToParentElement(options, selectEle);
        }
        // Set the default value of the <select>. Note that we escape this
        // value in quotes, just in case it contains a period or some other
        // character(s) that would mess up the querySelector.
        selectEle.querySelector(
            'option[value = "' + defaultVal + '"]'
        ).selected = true;
    }

    /* Given a list of element IDs and a boolean, changes the elements'
     * "disabled" attribute to false (if enable is true) and changes the
     * attribute to true if enable is false.
     *
     * ...So this just sets the disabled attribute to !enable.
     */
    function changeElementsEnabled(elements, enable) {
        for (var e = 0; e < elements.length; e++) {
            document.getElementById(elements[e]).disabled = !enable;
        }
    }

    /* Removes all of the child elements of an element.
     *
     * This function is based on
     * https://stackoverflow.com/a/3450726/10730311.
     * This way is apparently faster than just using
     * document.getElementById(divID).innerHTML = "".
     */
    function clearDiv(divID) {
        var element = document.getElementById(divID);
        while (element.firstChild) {
            element.removeChild(element.firstChild);
        }
    }

    /* Updates a <div> regarding how many samples have been dropped for a given
     * reason.
     *
     * numDroppedSamples: a list of sample IDs that have been dropped for
     * some reason.
     *  If the length of this is 0, then the <div> will be hidden.
     *  If the length of this is > 0, then the <div> will be un-hidden if
     *  it's currently hidden. (We define a "hidden" element as one that has
     *  the "invisible" CSS class.)
     *
     * totalSampleCount: an integer corresponding to the total number of
     * samples in this Qurro visualization.
     *  This will throw an error if totalSampleCount is 0, or if the number
     *  of dropped samples is greater than totalSampleCount.
     *
     * divID: the ID of the <div> we're updating
     *
     * dropType: This defines the reason we'll include in the <div> indicating
     * why samples have been dropped.
     *  If this is "balance", the reason will be "an undefined log-ratio."
     *  If this is "xAxis" or "color", the reason will be
     *  "a non-quantitative {f} field."
     *      ({f} will be replaced with whatever the optional field argument
     *      is.)
     */
    function updateSampleDroppedDiv(
        droppedSampleIDList,
        totalSampleCount,
        divID,
        dropType,
        field
    ) {
        var numDroppedSamples = droppedSampleIDList.length;
        validateSampleCounts(numDroppedSamples, totalSampleCount);

        // Only bother updating the <div>'s text if we're actually going to be
        // dropping samples for this "reason" -- i.e. numDroppedSamples > 0.
        if (numDroppedSamples > 0) {
            var prefix = "";
            if (dropType === "xAxis") {
                prefix = "x-axis: ";
            } else if (dropType === "color") {
                prefix = "Color: ";
            }

            // Figure out the reason we'll be displaying as a justification for
            // why at least this many samples have to be dropped.
            var reason = "(invalid reason given)";
            if (dropType === "balance") {
                reason = "an invalid (i.e. containing zero) log-ratio.";
            } else if (dropType === "xAxis" || dropType === "color") {
                reason = "an invalid " + field + " field.";
            }

            // We use textContent instead of innerHTML here because of the
            // reason variable, which includes field, which in turn could
            // conceivably include things like "</div>" that would mess up the
            // formatting.
            document.getElementById(divID).textContent =
                prefix +
                numDroppedSamples.toLocaleString() +
                " / " +
                totalSampleCount.toLocaleString() +
                " samples (" +
                formatPercentage(numDroppedSamples, totalSampleCount) +
                "%) can't be shown due to having " +
                reason;
            document.getElementById(divID).classList.remove("invisible");
        } else {
            document.getElementById(divID).classList.add("invisible");
        }
    }

    /* Given an object where each of the values is an array, computes the
     * union of all of these arrays and returns the length of the
     * union.
     *
     * Example usages:
     * unionSize({"a": [1,2,3], "b": [2,3,4,5]}) === 5
     * unionSize({"a": [1,2,3], "b": [4,5]}) === 5
     * unionSize({"a": [1,2], "b": [2,3,4,5], "c": [6]}) === 6
     * unionSize({"a": [], "b": [], "c": [6]}) === 1
     */
    function unionSize(mappingToArrays) {
        var keys = Object.keys(mappingToArrays);
        // Construct totalArray, which is just every array in mappingToArrays
        // concatenated. For the first example usage above, this would just be
        // something like [1,2,3,2,3,4,5].
        var totalArray = [];
        for (var k = 0; k < keys.length; k++) {
            totalArray = totalArray.concat(mappingToArrays[keys[k]]);
        }
        // Now that we have totalArray, we use vega.toSet() to convert it to a
        // mapping where each unique value in totalArray is a key. (See
        // https://vega.github.io/vega/docs/api/util/#toSet.) Taking the length
        // of the keys of this mapping gives us the "union size" we need.
        return Object.keys(vega.toSet(totalArray)).length;
    }

    function validateSampleCounts(droppedSampleCount, totalSampleCount) {
        if (totalSampleCount === 0) {
            throw new Error("# total samples cannot be 0");
        } else if (droppedSampleCount > totalSampleCount) {
            throw new Error("# dropped samples must be <= # total samples");
        }
    }

    /* Returns a string representation of the input value with two fractional
     * digits and formatted in the default locale.

     * This essentially mimics the behavior of percentage.toFixed(2), while
     * still respecting the user's locale. This solution is from
     * https://stackoverflow.com/a/31581206/10730311.
     */
    function formatPercentage(n, total) {
        var percentage = 100 * (n / total);
        return percentage.toLocaleString(undefined, {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        });
    }

    /* Updates a given <div> re: total # of samples shown.
     *
     * Sort of like the opposite of updateSampleDroppedDiv().
     *
     * Note that this will throw an error if totalSampleCount is 0 and/or if
     * the total number of dropped samples is greater than totalSampleCount.
     *
     * divID is an optional argument -- if not provided, it'll default to
     * "mainSamplesDroppedDiv".
     */
    function updateMainSampleShownDiv(droppedSamples, totalSampleCount, divID) {
        // compute union of all lists in droppedSamples. the length of
        // that is numSamplesShown.
        var droppedSampleCount = unionSize(droppedSamples);
        validateSampleCounts(droppedSampleCount, totalSampleCount);

        var numSamplesShown = totalSampleCount - unionSize(droppedSamples);
        var divIDInUse = divID === undefined ? "mainSamplesDroppedDiv" : divID;

        document.getElementById(divIDInUse).textContent =
            numSamplesShown.toLocaleString() +
            " / " +
            totalSampleCount.toLocaleString() +
            " samples (" +
            formatPercentage(numSamplesShown, totalSampleCount) +
            "%) currently shown.";
        // Just in case this div was set to invisible (i.e. this is the first
        // time it's been updated).
        document.getElementById(divIDInUse).classList.remove("invisible");
    }

    /* Downloads a string (either plain text or already a data URI) defining
     * the contents of a file.
     *
     * This is done by using a "downloadHelper" <a> tag.
     *
     * This function was based on downloadDataURI() in the MetagenomeScope
     * viewer interface source code.
     */
    function downloadDataURI(filename, contentToDownload, isPlainText) {
        document.getElementById("downloadHelper").download = filename;
        if (isPlainText) {
            var data =
                "data:text/plain;charset=utf-8;base64," +
                window.btoa(contentToDownload);
            document.getElementById("downloadHelper").href = data;
        } else {
            document.getElementById("downloadHelper").href = contentToDownload;
        }
        document.getElementById("downloadHelper").click();
    }

    /* If val is a string or number, this checks that val represents a valid,
     * finite numerical value (using vega.toNumber() and isFinite()). If so,
     * this returns that numerical value; otherwise, this returns NaN. (Also
     * returns NaN if val isn't a string or a number, although I don't think
     * this should ever happen in the course of regular usage of this
     * function.)
     *
     * This mimics how getInvalidSampleIDs() works (that is, in tandem with
     * Qurro's and QIIME 2's metadata readers). Input text is trimmed and then
     * attempted to be converted to a number using vega.toNumber(). Normally,
     * Number() (and therefore vega.toNumber()) has a silly corner case where
     * Number("   ") === 0. However, vega.toNumber("") === null, so using
     * .trim() on the input text (if a string) means that the output of
     * vega.toNumber() on the trimmed input will be null if the input text only
     * contains whitespace (and we can detect this and return NaN accordingly).
     *
     * This also treats Infinities/NaNs as invalid numbers, which matches the
     * sample metadata processing behavior.
     *
     * NOTE: Due to how numbers work in JS, precision here is inherently
     * limited. If you pass in, e.g,
     * "3.999999999999999999999999999999999999999999999999999999999", then
     * that'll get represented as 4 (but that'll happen even if you don't
     * represent it as a string, and this is also the case if you chuck that
     * same number into python).
     *
     * NOTE / TODO: If the number represented by val is outside of the range of
     * safe numbers allowed by JS (i.e. if val isn't in the range
     * [-(2^53 - 1), (2^53 - 1)]), this function will not work as you'd expect.
     * These problems are inherent to JS Numbers. It would be a good idea to
     * make this throw an error or something if this is the case, since we
     * could then let the user know the nature of what went wrong.
     * (Although using e.g. math.js is probably the best idea here.)
     */
    function getNumberIfValid(val) {
        if (typeof val === "string") {
            var nval = vega.toNumber(val.trim());
            if (nval !== null && isFinite(nval)) {
                return nval;
            }
        } else if (typeof val === "number") {
            if (isFinite(val)) {
                return val;
            }
        }
        return NaN;
    }

    // Array of all dropped-sample-statistics <div> IDs.
    // Used in a few places in the codebase, so I'm storing it here.
    var statDivs = [
        "mainSamplesDroppedDiv",
        "balanceSamplesDroppedDiv",
        "xAxisSamplesDroppedDiv",
        "colorSamplesDroppedDiv"
    ];

    return {
        setUpDOMBindings: setUpDOMBindings,
        populateSelect: populateSelect,
        changeElementsEnabled: changeElementsEnabled,
        clearDiv: clearDiv,
        updateSampleDroppedDiv: updateSampleDroppedDiv,
        unionSize: unionSize,
        updateMainSampleShownDiv: updateMainSampleShownDiv,
        formatPercentage: formatPercentage,
        downloadDataURI: downloadDataURI,
        getNumberIfValid: getNumberIfValid,
        statDivs: statDivs
    };
});
