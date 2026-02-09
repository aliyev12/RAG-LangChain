import type { Document } from "langchain";

export function combineDocuments(docs: Document[]): string {
  const pageContents = docs.map((doc) => doc.pageContent).join("\n\n");
  return pageContents;
}
