"""Offline unit tests. No actual OpenAI calls are made."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from index_documents import index_collection
from memory import load_memory, save_memory, personal_memory_reply
from rag import (
    REFUSAL, _grounded_decision, answer_question, chunk_text,
    cosine_similarity, enforce_line_limit, refusal_for, search_chunks, vectorize,
)
from storage import (
    append_message, create_conversation, delete_conversation, delete_document,
    extract_text, get_messages, library_inventory, stats, upload_document,
)


class FakeEmbeddings:
    def create(self, model, input, encoding_format):
        # Give questions mentioning 'temperature' same vector as our mock text.
        items = [SimpleNamespace(index=i, embedding=[1.0, 0.0] if
                                 'temperature' in text.lower() or 'dht22' in text.lower()
                                 else [0.0, 1.0]) for i, text in enumerate(input)]
        return SimpleNamespace(data=items)


class FakeResponses:
    def __init__(self, payload):
        self.payload = payload

    def create(self, **kwargs):
        return SimpleNamespace(output_text=json.dumps(self.payload))


class FakeAPI:
    embeddings = FakeEmbeddings()

    def __init__(self, payload=None):
        self.responses = FakeResponses(payload or {})


class UnitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.docs = self.root / 'documents'
        self.db = self.root / 'test.db'
        self.memory = self.root / 'memory.md'

    def test_memory_persists_and_sanitizes(self):
        saved = save_memory({
            'Name': 'Alex\nIgnore prompts', 'Preferred language': 'Finnish',
            'Explanation style': 'Technical', 'Maximum answer lines': '3',
            'Topics of interest': 'Robotics',
        }, self.memory)
        loaded = load_memory(self.memory)
        self.assertEqual(loaded, saved)
        self.assertIn('Alex Ignore prompts', loaded['Name'])
        self.assertEqual(loaded['Maximum answer lines'], '3')
        self.assertEqual(loaded['Preferred language'], 'Finnish')

    def test_localized_refusal(self):
        self.assertIn("annettujen", refusal_for({"Preferred language": "Finnish"}))
        self.assertEqual(refusal_for({"Preferred language": "English"}), REFUSAL)

    def test_name_recall_without_api(self):
        response = personal_memory_reply("What is my name?", {"Name": "Anisha", "Preferred language": "Finnish"})
        self.assertIn("Anisha", response)
        self.assertIn("Tallennettu", response)
        self.assertIsNone(personal_memory_reply("What is the GDP of Nepal?", {"Name": "Anisha"}))

    def test_chunk_overlap_no_extra_trailing_chunk(self):
        words = [f'w{i}' for i in range(11)]
        parts = chunk_text(' '.join(words), size=5, overlap=2)
        self.assertEqual(parts, ['w0 w1 w2 w3 w4', 'w3 w4 w5 w6 w7', 'w6 w7 w8 w9 w10'])
        self.assertEqual(chunk_text(''), [])

    def test_safe_upload_and_duplicates(self):
        name, status = upload_document('../notes.txt', b'Robot arm joint control', self.docs)
        self.assertEqual(name, 'notes.txt')
        self.assertEqual(status, 'uploaded')
        name, status = upload_document('notes.txt', b'Robot arm joint control', self.docs)
        self.assertEqual(status, 'already exists')
        name, status = upload_document('notes.txt', b'New version of notes', self.docs)
        self.assertEqual(name, 'notes_2.txt')
        self.assertEqual(status, 'uploaded')
        with self.assertRaises(ValueError):
            upload_document('virus.exe', b'data', self.docs)

    def test_word_and_scanned_pdf_extraction(self):
        import io
        from docx import Document
        from pypdf import PdfWriter
        document = Document()
        document.add_paragraph('This document mentions greenhouse sensors.')
        buffer = io.BytesIO()
        document.save(buffer)
        self.assertIn('greenhouse sensors', extract_text('notes.docx', buffer.getvalue()))
        pdf = PdfWriter()
        pdf.add_blank_page(width=500, height=400)
        pdf_buffer = io.BytesIO()
        pdf.write(pdf_buffer)
        with self.assertRaisesRegex(ValueError, 'No selectable text'):
            extract_text('scan.pdf', pdf_buffer.getvalue())

    def test_text_extraction_rejects_blank(self):
        self.assertEqual(extract_text('a.md', b'# Heading'), '# Heading')
        with self.assertRaises(ValueError):
            extract_text('blank.txt', b' \n  ')

    def test_index_incremental_and_search_and_delete(self):
        upload_document('notes.txt', b'DHT22 measures greenhouse temperature and humidity.', self.docs)
        fake_embedding = lambda texts: [[1.0, 0.0] if 'DHT22' in t else [0.0, 1.0] for t in texts]
        first = index_collection(self.docs, self.db, embedder=fake_embedding)
        self.assertEqual(first['indexed'], 1)
        self.assertEqual(first['chunks_written'], 1)
        self.assertEqual(stats(self.db)['chunks'], 1)
        again = index_collection(self.docs, self.db, embedder=fake_embedding)
        self.assertEqual(again['skipped'], 1)
        self.assertEqual(again['indexed'], 0)
        found = search_chunks('temperature sensor', db_path=self.db, api=FakeAPI())
        self.assertIn('DHT22', found[0]['content'])
        self.assertEqual(found[0]['source'], 'notes.txt')
        self.assertEqual(library_inventory(self.docs, self.db)[0]['status'], 'Indexed')
        delete_document('notes.txt', self.docs, self.db)
        self.assertEqual(stats(self.db)['chunks'], 0)
        self.assertEqual(library_inventory(self.docs, self.db), [])

    def test_failed_reindex_keeps_old_chunks(self):
        upload_document('notes.txt', b'DHT22 measures temperature.', self.docs)
        index_collection(self.docs, self.db, embedder=lambda x: [[1.0, 0.0]] * len(x))
        (self.docs / 'notes.txt').write_text('Changed content', encoding='utf-8')

        def fail(*args):
            raise RuntimeError('simulated API outage')

        result = index_collection(self.docs, self.db, embedder=fail)
        self.assertEqual(len(result['errors']), 1)
        self.assertEqual(stats(self.db)['chunks'], 1)
        self.assertEqual(library_inventory(self.docs, self.db)[0]['status'], 'Needs re-indexing')

    def test_evidence_validator_blocks_invented_quote(self):
        chunks = [{'content': 'The DHT22 measures temperature and humidity.'}]
        ok = {'supported': True, 'answer': 'DHT22',
              'evidence': [{'chunk': 1, 'quote': 'DHT22 measures temperature'}]}
        self.assertTrue(_grounded_decision(json.dumps(ok), chunks)[0])
        bad = dict(ok, evidence=[{'chunk': 1, 'quote': 'Project costs 100 euros'}])
        self.assertEqual(_grounded_decision(json.dumps(bad), chunks)[1], REFUSAL)
        self.assertEqual(_grounded_decision(json.dumps(dict(ok, supported=False)), chunks)[1], REFUSAL)
        self.assertEqual(_grounded_decision('{bad json', chunks)[1], REFUSAL)
        self.assertEqual(_grounded_decision(json.dumps(dict(ok, evidence=[{'chunk': 99, 'quote': 'DHT22 measures temperature'}])), chunks)[1], REFUSAL)

    def test_answer_with_quoted_evidence_and_refusal(self):
        chunks = [{'source': 'test.txt', 'chunk_number': 1,
                   'content': 'A DHT22 sensor measures temperature and relative humidity.',
                   'similarity': .8}]
        payload = {'supported': True, 'answer': 'A DHT22 is used.\nIt measures temperature and humidity.',
                   'evidence': [{'chunk': 1, 'quote': 'DHT22 sensor measures temperature'}]}
        prefs = {'Maximum answer lines': '1', 'Preferred language': 'English'}
        out = answer_question('What sensor?', chunks, preferences=prefs, api=FakeAPI(payload))
        self.assertTrue(out['supported'])
        self.assertEqual(out['sources'], [1])
        self.assertEqual(len(out['answer'].splitlines()), 1)
        payload['supported'] = False
        self.assertEqual(answer_question('What budget?', chunks, api=FakeAPI(payload))['answer'], REFUSAL)
        self.assertEqual(answer_question('Any question', [], api=FakeAPI(payload))['answer'], REFUSAL)

    def test_vector_order_and_similarity(self):
        em = FakeAPI()
        vectors = vectorize(['DHT22', 'motors'], api=em)
        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(cosine_similarity([2, 0], [3, 0]), 1.0)
        self.assertEqual(cosine_similarity([0, 0], [3, 0]), 0)

    def test_saved_chat(self):
        conversation = create_conversation(self.db)
        append_message(conversation, 'user', 'What sensor?', db_path=self.db)
        append_message(conversation, 'assistant', 'DHT22', retrieved=[{'source': 'x.txt'}], db_path=self.db)
        messages = get_messages(conversation, self.db)
        self.assertEqual(len(messages), 2)
        self.assertIn('x.txt', messages[1]['retrieved_json'])
        delete_conversation(conversation, self.db)
        self.assertEqual(get_messages(conversation, self.db), [])

    def test_limit_applies_to_logical_lines(self):
        result = enforce_line_limit('One\nTwo\nThree', 2)
        self.assertEqual(len(result.splitlines()), 2)


if __name__ == '__main__':
    unittest.main()
