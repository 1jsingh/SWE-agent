FROM sweagent/swe-agent:latest

COPY ../evaluation/evaluation.py /evaluation.py
RUN pip install git+https://github.com/1jsingh/SWE-bench.git
RUN pip install unidiff
CMD ["python", "/evaluation.py"]
