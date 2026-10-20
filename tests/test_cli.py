from document_reviewer import cli


def test_analyze_only_runs_analyzer_without_starting_workflow(monkeypatch, capsys) -> None:
    class Report:
        def model_dump_json(self, *, indent: int) -> str:
            assert indent == 2
            return '{"summary":"ok","issues":[]}'

    class AnalyzerOnlyAgents:
        def analyze(self, document: str) -> Report:
            assert document == "sample document"
            return Report()

    monkeypatch.setattr(cli, "read_document", lambda path: "sample document")
    monkeypatch.setattr(cli, "OllamaDocumentAgents", AnalyzerOnlyAgents)
    monkeypatch.setattr("sys.argv", ["reviewer", "--analyze-only", "sample.md"])

    cli.main()

    assert capsys.readouterr().out == '{"summary":"ok","issues":[]}\n'
