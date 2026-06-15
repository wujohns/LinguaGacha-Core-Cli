import path from "node:path";

import { ProjectDatabase } from "../database/database-operations";
import { ProjectSessionState } from "../project/project-session";
import { default_native_fs, type NativeFs } from "../../native/native-fs";
import {
  build_analysis_glossary_entries_from_candidates,
  type AnalysisCandidateGlossaryEntry,
} from "../../shared/analysis-candidate";
import type { ApiJsonValue } from "../api/api-types";
import { export_quality_rule_entries_to_files } from "../quality/quality-rule-file-io";
import * as AppErrors from "../../shared/error";

type MutableJsonRecord = Record<string, ApiJsonValue>;

export class AnalysisCandidateExportService {
  public constructor(
    private readonly database: ProjectDatabase,
    private readonly session_state: ProjectSessionState,
    private readonly native_fs: NativeFs = default_native_fs,
  ) {}

  public async export_analysis_candidates_to_directory(output_dir: string): Promise<{
    json_path: string;
    xlsx_path: string;
    entry_count: number;
  }> {
    const project_path = this.require_project_path();
    const output_base_path = path.join(path.resolve(output_dir), "glossary");
    this.native_fs.make_dir(path.dirname(output_base_path));
    const entries = this.build_entries(project_path);
    await export_quality_rule_entries_to_files(output_base_path, entries as unknown as MutableJsonRecord[]);
    return {
      json_path: `${output_base_path}.json`.replace(/\\/g, "/"),
      xlsx_path: `${output_base_path}.xlsx`.replace(/\\/g, "/"),
      entry_count: entries.length,
    };
  }

  private build_entries(project_path: string): AnalysisCandidateGlossaryEntry[] {
    const rows = this.database.execute({
      name: "getAnalysisCandidateAggregates",
      args: { projectPath: project_path },
    });
    const aggregate: Record<string, unknown> = {};
    if (Array.isArray(rows)) {
      for (const row of rows) {
        if (this.is_record(row)) {
          aggregate[String(row["src"] ?? "")] = row;
        }
      }
    }
    return build_analysis_glossary_entries_from_candidates(aggregate);
  }

  private require_project_path(): string {
    const state = this.session_state.snapshot();
    if (!state.loaded || state.projectPath === "") {
      throw new AppErrors.ProjectNotLoadedError();
    }
    return state.projectPath;
  }

  private is_record(value: unknown): value is MutableJsonRecord {
    return typeof value === "object" && value !== null && !Array.isArray(value);
  }
}
