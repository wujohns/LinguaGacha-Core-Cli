import { LANGUAGE_DISPLAY_NAMES } from "../../../../domain/language";

export const zh_cn_app = {
  metadata: {
    app_name: "LinguaGacha",
  },
  language: Object.fromEntries(
    Object.entries(LANGUAGE_DISPLAY_NAMES).map(([code, names]) => [code, names.zh]),
  ) as Record<keyof typeof LANGUAGE_DISPLAY_NAMES, string>,
  prompt: {
    builder_control_character_samples: "控制字符示例：",
    builder_glossary_header: "术语表 <术语原文> -> <术语译文> #<术语信息>:",
    builder_input: "输入：",
    builder_preceding_context: "参考上文：",
  },
  error: {
    request: {
      validation_failed: {
        message: "请求参数无效 …",
      },
      invalid_json: {
        message: "请求 JSON 无效 …",
      },
      route_not_found: {
        message: "API 路由不存在 …",
      },
    },
    project: {
      not_loaded: {
        message: "工程未加载 …",
        action: "请先打开或创建工程 …",
      },
      not_found: {
        message: "工程文件不存在 …",
        action: "请确认工程文件仍在原位置 …",
      },
    },
    file: {
      not_found: {
        message: "文件不存在 …",
        action: "请确认文件仍在原位置 …",
      },
      unsupported_format: {
        message: "不支持的文件格式 …",
        action: "请选择 LinguaGacha 支持的源文件 …",
      },
      parse_failed: {
        message: "文件内容解析失败 …",
        action: "请确认文件内容完整，或换用原始未损坏的文件 …",
      },
      invalid_structure: {
        message: "文件结构不符合格式要求 …",
        action: "请确认文件来源正确，或重新导出后再导入 …",
      },
      io_failed: {
        message: "文件读写失败 …",
      },
    },
    database: {
      conflict: {
        message: "数据库写入冲突，请刷新后重试 …",
        action: "请刷新当前数据后再次提交 …",
      },
    },
    data: {
      revision_conflict: {
        message: "数据版本已变化，请刷新后重试 …",
        action: "请刷新当前数据后再次提交 …",
      },
    },
    task: {
      busy: {
        message: "后台任务正在执行中，请稍后再试 …",
        action: "请等待当前任务结束或先停止任务 …",
      },
    },
    model: {
      not_found: {
        message: "模型配置不存在 …",
        action: "请重新选择模型配置 …",
      },
      provider_failed: {
        message: "模型服务请求失败，请检查接口配置 …",
        action: "请检查模型地址、密钥和服务商状态 …",
      },
    },
    worker: {
      failed: {
        message: "后台执行通道失败 …",
      },
      execution_failed: {
        message: "后台任务执行失败 …",
      },
    },
    runtime: {
      capability_missing: {
        message: "当前运行环境缺少必要能力 …",
      },
      disposed: {
        message: "运行资源已释放 …",
      },
      cancelled: {
        message: "操作已取消 …",
      },
      internal_invariant: {
        message: "内部状态异常 …",
      },
    },
    language: {
      invalid_target_language: {
        message: "目标语言无效 …",
      },
      unsupported_all_target_language: {
        message: "目标语言不支持全部语言 …",
      },
      unknown_source_language_code: {
        message: "源语言代码无效 …",
      },
    },
    quality: {
      unknown_rule_type: {
        message: "质量规则类型无效 …",
      },
      unsupported_rule_meta: {
        message: "质量规则配置项无效 …",
      },
    },
    prompt: {
      unknown_prompt_type: {
        message: "提示词类型无效 …",
      },
    },
  },
  diagnostic: {
    default_preset: {
      config_normalize_failed: "归一化默认预设配置失败：{CONFIG_PATH} …",
      prompt_load_failed: "默认提示词预设加载失败 …",
      quality_rule_load_failed: "默认质量规则预设加载失败 …",
      value_normalize_failed: "归一化默认预设值失败：{PRESET_DIRECTORY} -> {VALUE} …",
    },
    file_export: {
      open_output_folder_failed: "打开输出文件夹失败 …",
      translation_failed: "译文生成失败 …",
      write_file_failed: "文件写入失败 …",
    },
    lifecycle: {
      app_start_failed: "LinguaGacha 启动失败 …",
      backend_gateway_start_failed: "Backend 启动失败 …",
      main_fatal_uncaught: "运行时捕获到未处理致命异常 …",
    },
    migration: {
      path_failed: "迁移路径失败：{SOURCE_PATH} -> {DESTINATION_PATH} …",
    },
  },
  log: {
    api_test_fail: "接口测试失败 …",
    api_test_key: "正在测试密钥：",
    api_test_messages: "任务提示词：",
    api_test_result: "共测试 {COUNT} 个接口，成功 {SUCCESS} 个，失败 {FAILURE} 个 …",
    api_test_result_failure: "失败的密钥：",
    api_test_response_result: "模型回复内容：",
    api_test_timeout: "请求超时（{SECONDS} 秒）",
    api_test_token_info: "任务耗时 {TIME} 秒，输入消耗 {PT} Tokens，输出消耗 {CT} Tokens",
    app_version: "LinguaGacha v{VERSION} …",
    system_proxy_startup_detected: "检查到系统代理设置 - {PROXY}",
    default_preset_loaded: "已自动加载默认预设：{NAMES} …",
    engine_api_model: "接口模型",
    engine_api_name: "接口名称",
    engine_api_url: "接口地址",
    engine_task_done: "任务已完成 …",
    engine_task_exception: "任务执行失败 …",
    engine_task_fail: "任务未能全部完成，仍有部分数据未处理，请检查处理结果 …",
    engine_task_rule_analysis: "规则分析：",
    engine_task_thinking_process: "思考过程：",
    engine_task_stop: "任务已停止 …",
    engine_task_success:
      "任务耗时 {TIME} 秒，文本行数 {LINES} 行，输入消耗 {PT} Tokens，输出消耗 {CT} Tokens",
    generate_translation_done: "译文已保存至 {PATH} …",
    generate_translation_start: "生成译文中 …",
    response_checker_fail_data: "数据结构错误",
    response_checker_fail_degradation: "发生退化现象",
    response_checker_fail_line_count: "行数不一致",
    response_checker_fail_request: "模型请求失败",
    request_failed_retry: "模型请求失败，将自动重试 …",
    response_checker_fail_timeout: "网络请求超时",
    response_checker_line_error_empty_line: "存在空行",
    response_checker_line_error_hangeul: "谚文残留",
    response_checker_line_error_kana: "假名残留",
    response_checker_line_error_similarity: "较高相似度",
    system_closed_dropped: "日志系统已关闭，丢弃新日志：{MESSAGE}",
    translation_response_check_fail: "返回数据错误，将自动重试，原因：{REASON}",
    translation_response_check_fail_all: "全部译文质量校验失败，将自动切分重试，原因：{REASON}",
    translation_response_check_fail_part: "部分译文质量校验失败，将自动切分重试，原因：{REASON}",
    translation_task_result: "翻译结果：",
    translation_task_status_info:
      "拆分次数：{SPLIT} | 单条重试次数：{RETRY} | 任务长度阈值：{THRESHOLD}",
  },
} as const;
