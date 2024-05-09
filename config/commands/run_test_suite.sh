# @yaml
# signature: run_regression_safegaurd_tests
# docstring: runs and reports results on the regression tests, i.e. tests that check that normal usage (not related to the issue) still works before and after applying the fix. Note; these tests do not check if the specific issue is fixed, but are a safegaurd to ensure that the fix does not break normal usage. \n\n This should ideally return all tests passing before and after the fix.
run_regression_safegaurd_tests() {
    echo "<<REGRESSION_SAFEGAURD_TESTS||"
    echo "Running regression safegaurd tests..."
    # Run the regression safegaurd tests using the $TEST_CMD
    $TEST_CMD
    echo "||REGRESSION_SAFEGAURD_TESTS>>"
}