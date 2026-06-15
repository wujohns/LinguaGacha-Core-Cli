import { JsonTool } from "../../../../shared/utils/json-tool";

export class ResponseDecoder {
  public async decode_glossary_entries(response: string): Promise<Array<Record<string, string>>> {
    const glossary_entries: Array<Record<string, string>> = [];
    for (const line of response.split(/\r?\n/u)) {
      const stripped_line = line.trim();
      if (stripped_line === "" || stripped_line.startsWith("```")) {
        continue;
      }
      const json_data = await this.repair_parse_object(stripped_line);
      if (json_data === null) {
        continue;
      }
      const glossary_entry = this.build_glossary_entry(json_data);
      if (glossary_entry !== null) {
        glossary_entries.push(glossary_entry);
      }
    }
    return glossary_entries;
  }

  private build_glossary_entry(json_data: Record<string, unknown>): Record<string, string> | null {
    if (Object.keys(json_data).length !== 3) {
      return null;
    }
    if (!("src" in json_data) || !("dst" in json_data) || !("type" in json_data)) {
      return null;
    }
    return {
      src: typeof json_data.src === "string" ? json_data.src : "",
      dst: typeof json_data.dst === "string" ? json_data.dst : "",
      info: typeof json_data.type === "string" ? json_data.type : "",
    };
  }

  private async repair_parse_object(text: string): Promise<Record<string, unknown> | null> {
    try {
      const value = await JsonTool.repairParse<unknown>(text);
      return typeof value === "object" && value !== null && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : null;
    } catch {
      return null;
    }
  }
}
