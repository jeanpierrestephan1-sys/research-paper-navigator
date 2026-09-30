import Foundation
import PDFKit

let dir = CommandLine.arguments[1]
let fm = FileManager.default
let files = (try! fm.contentsOfDirectory(atPath: dir)).filter { $0.hasSuffix(".pdf") }.sorted()

func titleOf(_ path: String) -> String {
    guard let doc = PDFDocument(url: URL(fileURLWithPath: path)),
          let page = doc.page(at: 0), let s = page.string else { return "?? unreadable" }
    var lines = s.components(separatedBy: .newlines)
        .map { $0.trimmingCharacters(in: .whitespaces) }
        .filter { !$0.isEmpty }
    // drop arXiv banners, page furniture
    lines = lines.filter {
        !$0.lowercased().hasPrefix("arxiv:") &&
        !$0.lowercased().hasPrefix("published as") &&
        !$0.lowercased().hasPrefix("proceedings of") &&
        $0.count > 2
    }
    var title = ""
    for l in lines.prefix(4) {
        // stop once we hit an author/affiliation/email line
        if l.contains("@") || l.lowercased().contains("university")
            || l.lowercased().contains("institute") || l.lowercased().hasPrefix("abstract") { break }
        title += (title.isEmpty ? "" : " ") + l
        if title.count > 40 && (l.hasSuffix("?") || l.hasSuffix(".")) { break }
    }
    if title.isEmpty { title = lines.first ?? "??" }
    // collapse hyphenation and whitespace
    title = title.replacingOccurrences(of: "- ", with: "")
    return title.trimmingCharacters(in: .whitespaces)
}

for f in files {
    let t = titleOf(dir + "/" + f)
    print("\(f)\t\(t)")
}
