import Foundation
import PDFKit
let args = CommandLine.arguments
guard args.count > 1, let doc = PDFDocument(url: URL(fileURLWithPath: args[1])) else {
    FileHandle.standardError.write("cannot open\n".data(using:.utf8)!); exit(1)
}
let maxPages = args.count > 2 ? Int(args[2])! : doc.pageCount
for i in 0..<min(maxPages, doc.pageCount) {
    if let p = doc.page(at: i), let s = p.string { print(s) }
}
