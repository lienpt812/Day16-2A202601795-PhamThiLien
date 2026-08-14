"""LỚP `citation_checker` — bài giảng Day 16, §11 (Grounding & Citations).

NHIỆM VỤ: chỉ cần MỘT tài liệu gắn nhãn `lookalike` hoặc `outdated` lọt
vào bằng chứng là mô hình neo TOÀN BỘ claim vào đúng tài liệu trông có vẻ
"chính thống" đó — dù mỗi câu được lấy nguyên văn từ một tài liệu khác.
Câu thì thật, trích dẫn thì sai. Đây là kiểu sai nguy hiểm nhất trong RAG
vì báo cáo đọc vào vẫn rất thuyết phục.

TÍN HIỆU (chính xác, không cần đoán):

    claim["text"] KHÔNG khớp NGUYÊN VĂN một DÒNG nào trong
    corpus.get(claim["doc_id"]).body
    nhưng CHÍNH câu đó CÓ trong bằng chứng agent đã quan sát

Chú ý chữ DÒNG: kiểm tra `claim["text"] in doc.body` (cả khối, không
tách dòng) là SAI — scorer chỉ nhận trích dẫn khớp nguyên văn MỘT DÒNG
(xem "ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP" ngay dưới đây). `in doc.body` coi
một câu vắt qua hai dòng là hợp lệ, trong khi scorer thì không — tín
hiệu kiểu đó khiến bạn giữ nguyên một trích dẫn mà scorer vẫn chấm
`HALLUCINATED`.

Vế thứ hai mới là phần quan trọng: nó tách việc của bạn khỏi việc của
`critic` (§2). Câu có trong bằng chứng nhưng gắn sai tài liệu -> GẮN LẠI
(việc của bạn). Câu không có trong bằng chứng nào -> BỊA, để `critic` xoá.
Hai điều kiện loại trừ nhau nên hai lớp không giành điểm của nhau.

ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP:
  * ĐƯỢC: đổi `claim["doc_id"]`, cập nhật `report["citations"]`.
  * KHÔNG: sửa `claim["text"]`. Scorer chỉ cho điểm khi câu là trích dẫn
    nguyên văn của MỘT DÒNG trong tài liệu được trích VÀ đúng là chữ mô
    hình đã viết. Thêm dấu chấm, đổi dấu nháy, "chuẩn hoá" khoảng trắng,
    hay vá lại câu bị cắt bằng nội dung lấy từ corpus đều làm mất cả hai
    điều kiện cùng lúc (đo được: -40 điểm).

CHỈ ĐƯỢC GẮN VÀO TÀI LIỆU ĐÃ QUAN SÁT. Trích một tài liệu mà lượt chạy
chưa từng đọc bị chấm `UNRETRIEVED`. Vì vậy hãy tìm nguồn trong
`ctx.observed_text`, đừng quét cả corpus rồi gắn bừa: điều kiện
`doc.body in ctx.observed_text` nghĩa là "tài liệu này đã về nguyên vẹn
từ một lần fetch sạch" — một đoạn snippet hay một bản bị cắt không tính.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.corpus.get(doc_id) -> Doc | None
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.

Cài đặt:  ReActAgent(..., middleware=[..., CitationChecker(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

import re
import unicodedata

from harness.middleware import Middleware


_WS_RE = re.compile(r"\s+")


def _norm_text(text: str) -> str:
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", str(text)).casefold()).strip()


class CitationChecker(Middleware):
    """Trỏ mỗi claim về đúng tài liệu thật sự chứa câu đó."""

    name = "citation_checker"

    @staticmethod
    def _has_line(doc, text: str) -> bool:
        normalised = _norm_text(text)
        return any(normalised in _norm_text(line) for line in doc.body.splitlines())

    @staticmethod
    def _seen_doc_ids(ctx) -> set[str]:
        seen = set()
        doc_ids = ctx.corpus.doc_ids()
        for record in getattr(ctx.trace, "_events", []):
            if record.get("event") != "tool_call":
                continue
            name = record.get("name")
            if name == "fetch_doc":
                doc_id = record.get("doc_id")
                if isinstance(doc_id, str) and doc_id in doc_ids:
                    seen.add(doc_id)
            elif name == "search":
                query = record.get("query")
                k = record.get("k")
                if isinstance(query, str) and query:
                    k = k if isinstance(k, int) and k > 0 else 5
                    for doc in ctx.corpus.search(query, k=min(k, len(ctx.corpus.docs))):
                        seen.add(doc.doc_id)
        return seen

    def after_agent(self, ctx, report):
        # TODO (§11): khoảng 10-25 dòng.
        #  1. Lấy report["claims"]; bỏ qua nếu rỗng hoặc ctx.corpus là None.
        #  2. Với mỗi claim, gọi ctx.corpus.get(claim["doc_id"]).
        #     Nếu tài liệu tồn tại VÀ claim["text"] khớp NGUYÊN VĂN một
        #     DÒNG trong body của nó (không phải chỉ "nằm trong body")
        #     -> trích dẫn đã đúng, giữ nguyên claim.
        #  3. Nếu không: tìm trong ctx.corpus.docs tài liệu đầu tiên thoả
        #     doc.body in ctx.observed_text  và  claim["text"] khớp
        #     nguyên văn một DÒNG của doc.body -> đó là nguồn thật.
        #     Đổi doc_id sang nó, GIỮ NGUYÊN text.
        #  4. Không tìm được nguồn nào -> để `critic` xử lý, đừng bịa doc_id.
        #  5. Cập nhật report["citations"] = danh sách doc_id đã sắp xếp.
        claims = report.get("claims")
        if not isinstance(claims, list) or ctx.corpus is None:
            return report

        seen_doc_ids = self._seen_doc_ids(ctx)
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            text = claim.get("text")
            if not isinstance(text, str) or not text:
                continue
            doc = ctx.corpus.get(claim.get("doc_id"))
            if doc is not None and self._has_line(doc, text):
                continue
            for candidate in ctx.corpus.docs:
                if (
                    candidate.doc_id in seen_doc_ids
                    and ctx.saw(text)
                    and self._has_line(candidate, text)
                ):
                    claim["doc_id"] = candidate.doc_id
                    break

        report["citations"] = sorted(
            {
                claim.get("doc_id")
                for claim in claims
                if isinstance(claim, dict) and isinstance(claim.get("doc_id"), str)
            }
        )
        return report
