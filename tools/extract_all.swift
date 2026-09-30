// Bulk PDF -> text extraction for the capstone corpus.
// Usage: swift extract_all.swift ResearchPapers text-cache
// Writes one .txt per .pdf, page-delimited so chunks can cite a page number.
import Foundation
import PDFKit

let src = CommandLine.arguments[1]
let dst = CommandLine.arguments[2]
let fm = FileManager.default
try? fm.createDirectory(atPath: dst, withIntermediateDirectories: true)

let files = (try! fm.contentsOfDirectory(atPath: src)).filter { $0.hasSuffix(".pdf") }.sorted()
var done = 0
for f in files {
    let out = dst + "/" + f.replacingOccurrences(of: ".pdf", with: ".txt")
    if fm.fileExists(atPath: out) { done += 1; continue }
    guard let doc = PDFDocument(url: URL(fileURLWithPath: src + "/" + f)) else {
        FileHandle.standardError.write("UNREADABLE \(f)\n".data(using: .utf8)!)
        continue
    }
    var buf = ""
    for i in 0..<doc.pageCount {
        if let s = doc.page(at: i)?.string {
            buf += "\u{000C}PAGE \(i + 1)\n" + s + "\n"
        }
    }
    try? buf.write(toFile: out, atomically: true, encoding: .utf8)
    done += 1
    FileHandle.standardError.write("[\(done)/\(files.count)] \(f) — \(doc.pageCount)p, \(buf.count)c\n".data(using: .utf8)!)
}
FileHandle.standardError.write("done: \(done)/\(files.count)\n".data(using: .utf8)!)
