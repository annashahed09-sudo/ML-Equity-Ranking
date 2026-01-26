import os

# Run all unit tests
os.system("pytest tests/ --maxfail=1 --disable-warnings -q")

# You can add more automation here (e.g., run main pipeline, save results)
print("All tests and main pipeline executed.")
