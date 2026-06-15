import { en_us_app } from "./resources/en-US/app";
import { zh_cn_app } from "./resources/zh-CN/app";
import type { LocaleMessageSchema } from "./types";

export const zh_cn_messages = {
  app: zh_cn_app,
} as const;

export const en_us_messages = {
  app: en_us_app,
} satisfies LocaleMessageSchema<typeof zh_cn_messages>;
