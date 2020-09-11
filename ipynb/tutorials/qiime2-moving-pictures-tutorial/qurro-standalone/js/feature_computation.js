define(["./dom_utils"], function(dom_utils) {
    /* Converts a feature field value to a text-searchable value, if possible.
     *
     * If the input is a string, returns the input (in lower case).
     * If the input is a number, returns the input as a string.
     *     (This is useful if users want to search numbers as text, which is
     *     kind of silly but still valid.)
     * If the input is neither of those types, returns null to indicate that we
     * can't "search" the input.
     */
    function tryTextSearchable(fmVal) {
        if (typeof fmVal === "string") {
            return fmVal.toLowerCase();
        } else if (typeof fmVal === "number") {
            return String(fmVal);
        } else {
            return null;
        }
    }

    /* Given a list of feature "rows", a string of input text, and a feature
     * field, returns a list of all features that *do* or *do not* contain that
     * text in the specified feature field.
     *
     * If "negate" is truthy, then this will return a list of all features that
     * *do not* contain the specified text in the specified feature field.
     * Otherwise, this will return a list of all features that *do* contain the
     * specified text in the specified feature field.
     *
     * Note that this can lead to some weird results if you're not careful --
     * e.g. just searching on "Staphylococcus" will include Staph phages in the
     * filtering (since their names contain the text "Staphylococcus").
     */
    function textFilterFeatures(
        featureRowList,
        inputText,
        featureField,
        negate
    ) {
        var filteredFeatures = [];
        var currVal;
        var decisionFunc;
        if (negate) {
            decisionFunc = function(value) {
                return !value.includes(inputText);
            };
        } else {
            decisionFunc = function(value) {
                return value.includes(inputText);
            };
        }
        for (var ti = 0; ti < featureRowList.length; ti++) {
            currVal = tryTextSearchable(featureRowList[ti][featureField]);
            if (currVal === null) {
                continue;
            } else if (decisionFunc(currVal)) {
                filteredFeatures.push(featureRowList[ti]);
            }
        }
        return filteredFeatures;
    }

    /* Given an operator ("lt", "gt", "lte", or "gte"), returns a comparison
     * function that takes in a single number (i) as input and returns true if:
     *
     * "lt":  i <  n
     * "gt":  i >  n
     * "lte": i <= n
     * "gte": i >= n
     *
     * Throws an error if operator isn't one of the four possible values listed
     * above.
     */
    function operatorToCompareFunc(operator, n) {
        if (operator === "lt") {
            return function(i) {
                return i < n;
            };
        } else if (operator === "gt") {
            return function(i) {
                return i > n;
            };
        } else if (operator === "lte") {
            return function(i) {
                return i <= n;
            };
        } else if (operator === "gte") {
            return function(i) {
                return i >= n;
            };
        } else {
            throw new Error(
                "unrecognized operator passed; must be 'lt', " +
                    "'gt', 'lte', or 'gte'"
            );
        }
    }

    /* Given a list of feature "rows", a number, a feature field, and an
     * "operator" string, returns a list of all features where the feature's
     * field value is both numeric and compares to the input number properly.
     *
     * Valid values for "operator" are "lt", "gt", "lte", and "gte"
     * (corresponding to the comparison operators <, >, <=, and >=). Passing
     * anything else for the "operator" argument will result in an error being
     * thrown by operatorToCompareFunc().
     *
     * As an example: if the input features' field values are "asdf",
     * 3, 5, and 10, the inputNum is 6, and the operator is "lt", then this
     * will return the features with field values of 3 and 5.
     */
    function numberBasicFilterFeatures(
        featureRowList,
        inputNum,
        featureField,
        operator
    ) {
        // Get a comparison function based on the operator and inputNum
        var compareFunc = operatorToCompareFunc(operator, inputNum);
        var filteredFeatures = [];
        var currNum, currVal;

        // Loop through every feature, checking the particular feature field's
        // values accordingly
        for (var ti = 0; ti < featureRowList.length; ti++) {
            currVal = featureRowList[ti][featureField];
            // This check is basically equivalent to what
            // RRVDisplay.getInvalidSampleIDs() does. For both sample and
            // feature metadata values, we know that the input is either a
            // string/number or a null value.
            // If this feature's field value is null, or if it isn't a valid
            // numerical value, then we won't bother calling compareFunc on it.
            // Otherwise, we'll try that.
            if (currVal === null) {
                continue;
            } else {
                // currNum will either be a normal number or NaN, so we can
                // just test its validity with !isNaN().
                currNum = dom_utils.getNumberIfValid(currVal);
                // Only do the comparison if currNum isn't NaN (uses boolean
                // short-circuiting).
                if (!isNaN(currNum) && compareFunc(currNum)) {
                    filteredFeatures.push(featureRowList[ti]);
                }
            }
        }
        return filteredFeatures;
    }

    /* Prepares an array of separated text fragments (also referred to as
     * "ranks"), either from the input text or from a feature field.
     *
     * In rank searching, users can search for multiple ranks at once if they
     * separate them with a comma, a semicolon, or a space; and we rely on
     * these characters not being present within a taxonomic rank (as far as I
     * can tell, all standard formats use semicolons or commas as separators).
     *
     * The choice to consider spaces as a "separator" within taxonomic ranks
     * might be a bit idiosyncratic. We can change this in the future if
     * needed.
     */
    function textToRankArray(text) {
        if (typeof text !== "string") {
            return [];
        }
        var initialRankArray = text
            .trim()
            .replace(/[,;\s]/g, " ")
            .split(" ");
        // Filter out ""s caused by repeated commas or whitespace in the input.
        // Why we need this: "a b   c".split(" ") produces
        // ["a", "b", "", "", "c"] and we just want ["a", "b", "c"]
        return initialRankArray.filter(function(r) {
            return r !== "";
        });
    }

    /* Returns true if arrayA and arrayB share at least one element.
     *
     * The check is done via === ("strict equality" in JS).
     *
     * Since this returns as soon as a match is found, this should be pretty
     * efficient.
     */
    function existsIntersection(arrayA, arrayB) {
        for (var a = 0; a < arrayA.length; a++) {
            for (var b = 0; b < arrayB.length; b++) {
                if (arrayA[a] === arrayB[b]) {
                    // If we found a match, no need to keep checking.
                    return true;
                }
            }
        }
        return false;
    }

    /* Given a list of feature "rows", a string of input "ranks," and a feature
     * field, returns a list of all features that contain a taxonomic
     * rank that matches a rank in the input. (The input(s) and things being
     * searched for don't actually have to refer to taxonomic ranks, but this
     * functionality was designed for use with taxonomy strings -- the problem
     * it addresses is when one taxonomic rank contains another rank's name,
     * e.g. "Staphylococcus_phage" showing up in normal search results for
     * "Staphylococcus".)
     *
     * First, we throw the input text through textToRankArray() above to
     * get a list of separated text fragments in the input.
     *
     * Next, we go through the features one-by-one. Each feature's value for
     * the specified feature field will be split up using
     * textToRankArray(). We then search for exact matches (not just
     * "does this contain the input text," like in textFilterFeatures(), but
     * "is this exactly equal to the input text?"), and return a list of
     * all features where at least one separated text fragment matched the
     * input text fragment(s).
     */
    function rankFilterFeatures(featureRowList, inputText, featureField) {
        var inputRankArray = textToRankArray(inputText);
        if (inputRankArray.length <= 0) {
            return [];
        }
        var ranksOfFeatureMetadata;
        var filteredFeatures = [];
        for (var ti = 0; ti < featureRowList.length; ti++) {
            // If the current feature field value is null / otherwise not
            // text-searchable, tryTextSearchable() returns null (which will cause
            // textToRankArray() to return [], which will cause
            // existsIntersection() to return false quickly).
            ranksOfFeatureMetadata = textToRankArray(
                tryTextSearchable(featureRowList[ti][featureField])
            );
            if (existsIntersection(ranksOfFeatureMetadata, inputRankArray)) {
                filteredFeatures.push(featureRowList[ti]);
            }
        }
        return filteredFeatures;
    }

    /* Returns list of feature data objects (in the rank plot JSON) based
     * on some sort of "match" of a given feature metadata/ranking field
     * (including Feature ID) with the input text. The input text must be a
     * string (even numbers aren't allowed -- the conversion is done later on
     * in this function if a numeric search type is being used).
     *
     * If inputText is empty (i.e. its length is 0), this returns an empty
     * array.
     *
     * Will raise an error if the featureField isn't present in the data or if
     * the searchType is unrecognized. (The filtering functions this delegates
     * to may also raise errors.)
     */
    function filterFeatures(rankPlotJSON, inputText, featureField, searchType) {
        if (
            featureField !== "Feature ID" &&
            rankPlotJSON.datasets.qurro_feature_metadata_ordering.indexOf(
                featureField
            ) < 0 &&
            rankPlotJSON.datasets.qurro_rank_ordering.indexOf(featureField) < 0
        ) {
            throw new Error("featureField not found in data");
        } else if (inputText.length === 0) {
            return [];
        }

        var potentialFeatures = rankPlotJSON.datasets[rankPlotJSON.data.name];
        var inputNum;
        if (searchType === "rank") {
            return rankFilterFeatures(
                potentialFeatures,
                inputText.toLowerCase(),
                featureField
            );
        } else if (searchType === "text" || searchType === "nottext") {
            var negate = searchType === "nottext";
            return textFilterFeatures(
                potentialFeatures,
                inputText.toLowerCase(),
                featureField,
                negate
            );
        } else if (
            searchType === "lt" ||
            searchType === "gt" ||
            searchType === "lte" ||
            searchType === "gte"
        ) {
            inputNum = dom_utils.getNumberIfValid(inputText);
            if (isNaN(inputNum)) {
                return [];
            }
            return numberBasicFilterFeatures(
                potentialFeatures,
                inputNum,
                featureField,
                searchType
            );
        } else if (
            searchType === "autoPercentTop" ||
            searchType === "autoPercentBot" ||
            searchType === "autoLiteralTop" ||
            searchType === "autoLiteralBot"
        ) {
            var inPercentages = searchType.startsWith("autoPercent");
            var featureCt = potentialFeatures.length;
            inputNum = dom_utils.getNumberIfValid(inputText);
            // Initial check for validity: regardless of if inputNum describes
            // a "literal" or "percentage"-based cutoff, it should be a finite
            // number that's at least 0
            if (isNaN(inputNum) || inputNum < 0) {
                return [];
            }
            // If the user asks for more than 100% of the features (aka more
            // features than are present in the dataset), just go ahead and
            // return all features
            else if (
                (inPercentages && inputNum > 100) ||
                (!inPercentages && inputNum > featureCt)
            ) {
                return potentialFeatures;
            }
            // OK, so now we know that inputNum is valid (i.e. is a number and
            // is in the range [0, 100] for percentage searching or [0, #
            // features] for number-of-features searching).
            // Next, let's just figure out how many features to extract from a
            // given side of the ranking.
            var numberOfFeaturesToGet;
            if (inPercentages) {
                // Why floor? If the user requests, say, the top and bottom
                // 33.33% of features, and there are 10 features, then it makes
                // more sense to give 3 features on each side than 4 (IMO).
                numberOfFeaturesToGet = Math.floor(
                    (inputNum / 100) * featureCt
                );
            } else {
                // If inputNum is a float, we just take the floor of it (so if
                // the user asks for the top/bottom 43.7 features we'll just
                // return the top/bottom 43 features).
                // We could also reject float values above, but I don't think
                // that'd be super user-friendly to people going between % and
                // "literal # of feature" modes.
                numberOfFeaturesToGet = Math.floor(inputNum);
            }
            var useTop = searchType.endsWith("Top");
            return extremeFilterFeatures(
                potentialFeatures,
                numberOfFeaturesToGet,
                featureField,
                useTop
            );
        } else {
            throw new Error("unrecognized searchType");
        }
    }

    /* Returns list of "n" feature data objects from either the top or bottom
     * side of the feature rankings.
     *
     * featureRowList is a list of feature rows (same as the other
     * *FilterFeatures() methods), n is an integer, ranking is a feature
     * ranking shared by every feature in featureRowList, and useTop is a
     * boolean value.
     *
     * Throws an error if any features don't have the specified ranking.
     */
    function extremeFilterFeatures(featureRowList, n, ranking, useTop) {
        // Sort features by the specified ranking in featureRowList
        var sortedFeatureRowList = featureRowList.sort(
            // Compare features by their "ranking field" values, i.e. the
            // literal differential or feature loading values.
            // (...These should all explicitly be numbers, as guaranteed by our
            // use of pd.to_numeric() in qurro.generate.gen_rank_plot().)
            // Comparison function based on spec here (thanks MDN!) --
            // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/sort#Description
            function(feature1, feature2) {
                var f1r = feature1[ranking];
                var f2r = feature2[ranking];
                // Basic validation to ensure that both features have this
                // ranking, and that it isn't null or whatever (should never
                // happen in practice due to validation on the python side of
                // things, but might as well be careful)
                if (typeof f1r !== "number" || typeof f2r !== "number") {
                    throw new Error(
                        ranking +
                            " ranking not present and/or numeric for all features"
                    );
                }
                if (f1r < f2r) {
                    return -1;
                } else if (f1r > f2r) {
                    return 1;
                } else {
                    return 0;
                }
            }
        );
        var featureCt = featureRowList.length;
        if (useTop) {
            // get top n features for the given ranking
            return sortedFeatureRowList.slice(featureCt - n);
        } else {
            // get bottom n features for the given ranking
            return sortedFeatureRowList.slice(0, n);
        }
    }

    /* We set the balance for samples with an abundance of <= 0 in either
     * the top or bottom of the log-ratio as null.
     *
     * RRVDisplay.updateSamplePlotFilters() should ensure that samples with
     * a null log-ratio are filtered out of the sample plot.
     */
    function computeBalance(topValue, botValue) {
        if (typeof topValue !== "number" || typeof botValue !== "number") {
            throw new Error(
                "computeBalance() called with non-numerical input(s)"
            );
        }
        if (topValue <= 0 || botValue <= 0) {
            return null;
        }
        return Math.log(topValue) - Math.log(botValue);
    }

    return {
        filterFeatures: filterFeatures,
        extremeFilterFeatures: extremeFilterFeatures,
        computeBalance: computeBalance,
        textToRankArray: textToRankArray,
        operatorToCompareFunc: operatorToCompareFunc,
        existsIntersection: existsIntersection,
        tryTextSearchable: tryTextSearchable
    };
});
