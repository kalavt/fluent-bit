/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*  Fluent Bit
 *  ==========
 *  Copyright (C) 2015-2024 The Fluent Bit Authors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>
#include <parquet/exception.h>

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

extern "C" {
#include <fluent-bit/flb_macros.h>
#include <fluent-bit/flb_log.h>
#include <fluent-bit/flb_mem.h>
#include <fluent-bit/flb_error.h>
#include <fluent-bit/flb_sds.h>
#include <fluent-bit/flb_parquet.h>
#include <msgpack.h>
#include <cJSON.h>
}

namespace {

// Constants for buffer sizes
constexpr size_t INITIAL_PARQUET_BUFFER_SIZE = 16 * 1024;  // 16KB initial buffer for Arrow
constexpr size_t ESTIMATED_RECORD_SIZE = 256;              // Estimate for vector pre-allocation

// Convert msgpack object to Arrow array builders
class MsgpackToArrowConverter {
public:
    /* Statistics for overflow/clamping operations - per field */
    std::unordered_map<std::string, size_t> int32_overflow_by_field;
    std::unordered_map<std::string, size_t> int64_overflow_by_field;
    std::unordered_map<std::string, size_t> float_to_int_clamp_by_field;

    arrow::Status convert_value(const msgpack_object* obj,
                                arrow::ArrayBuilder* builder,
                                const std::shared_ptr<arrow::DataType>& type,
                                const std::string& field_name) {

        // Handle null values
        if (obj->type == MSGPACK_OBJECT_NIL) {
            return builder->AppendNull();
        }

        switch (type->id()) {
            case arrow::Type::BOOL:
                return convert_to_bool(obj, static_cast<arrow::BooleanBuilder*>(builder));

            case arrow::Type::INT32:
                return convert_to_int32(obj, static_cast<arrow::Int32Builder*>(builder), field_name);

            case arrow::Type::INT64:
                return convert_to_int64(obj, static_cast<arrow::Int64Builder*>(builder), field_name);

            case arrow::Type::FLOAT:
                return convert_to_float(obj, static_cast<arrow::FloatBuilder*>(builder));

            case arrow::Type::DOUBLE:
                return convert_to_double(obj, static_cast<arrow::DoubleBuilder*>(builder));

            case arrow::Type::STRING:
                return convert_to_string(obj, static_cast<arrow::StringBuilder*>(builder));

            case arrow::Type::BINARY:
                return convert_to_binary(obj, static_cast<arrow::BinaryBuilder*>(builder));

            case arrow::Type::TIMESTAMP:
                return convert_to_timestamp(obj, static_cast<arrow::TimestampBuilder*>(builder));

            default:
                return arrow::Status::NotImplemented(
                    "Unsupported Arrow type: " + type->ToString());
        }
    }

private:
    arrow::Status convert_to_bool(const msgpack_object* obj, arrow::BooleanBuilder* builder) {
        if (obj->type == MSGPACK_OBJECT_BOOLEAN) {
            return builder->Append(obj->via.boolean);
        } else if (obj->type == MSGPACK_OBJECT_POSITIVE_INTEGER) {
            return builder->Append(obj->via.u64 != 0);
        } else if (obj->type == MSGPACK_OBJECT_NEGATIVE_INTEGER) {
            return builder->Append(obj->via.i64 != 0);
        } else {
            /* Return error instead of NULL for type mismatch - let caller decide */
            return arrow::Status::Invalid("Cannot convert msgpack type to bool");
        }
    }

    arrow::Status convert_to_int32(const msgpack_object* obj, arrow::Int32Builder* builder,
                                   const std::string& field_name) {
        if (obj->type == MSGPACK_OBJECT_POSITIVE_INTEGER) {
            /* Check for overflow: uint64 → int32 */
            if (obj->via.u64 > INT32_MAX) {
                int32_overflow_by_field[field_name]++;
                return builder->Append(INT32_MAX);
            }
            return builder->Append(static_cast<int32_t>(obj->via.u64));
        } else if (obj->type == MSGPACK_OBJECT_NEGATIVE_INTEGER) {
            /* Check for underflow: int64 → int32 */
            if (obj->via.i64 < INT32_MIN || obj->via.i64 > INT32_MAX) {
                int32_overflow_by_field[field_name]++;
                return builder->Append(obj->via.i64 < INT32_MIN ? INT32_MIN : INT32_MAX);
            }
            return builder->Append(static_cast<int32_t>(obj->via.i64));
        } else if (obj->type == MSGPACK_OBJECT_FLOAT32 || obj->type == MSGPACK_OBJECT_FLOAT64) {
            /* Allow float to int32 conversion with clamping (common use case) */
            if (obj->via.f64 > INT32_MAX || obj->via.f64 < INT32_MIN) {
                float_to_int_clamp_by_field[field_name]++;
                return builder->Append(obj->via.f64 > INT32_MAX ? INT32_MAX : INT32_MIN);
            }
            return builder->Append(static_cast<int32_t>(obj->via.f64));
        } else {
            /* Return error instead of NULL for type mismatch */
            return arrow::Status::Invalid("Cannot convert msgpack type to int32");
        }
    }

    arrow::Status convert_to_int64(const msgpack_object* obj, arrow::Int64Builder* builder,
                                   const std::string& field_name) {
        if (obj->type == MSGPACK_OBJECT_POSITIVE_INTEGER) {
            /* Check for overflow: uint64 → int64 */
            if (obj->via.u64 > static_cast<uint64_t>(INT64_MAX)) {
                int64_overflow_by_field[field_name]++;
                return builder->Append(INT64_MAX);
            }
            return builder->Append(static_cast<int64_t>(obj->via.u64));
        } else if (obj->type == MSGPACK_OBJECT_NEGATIVE_INTEGER) {
            return builder->Append(obj->via.i64);
        } else if (obj->type == MSGPACK_OBJECT_FLOAT32 || obj->type == MSGPACK_OBJECT_FLOAT64) {
            return builder->Append(static_cast<int64_t>(obj->via.f64));
        } else if (obj->type == MSGPACK_OBJECT_BOOLEAN) {
            return builder->Append(obj->via.boolean ? 1 : 0);
        } else {
            /* Return error instead of NULL for type mismatch */
            return arrow::Status::Invalid("Cannot convert msgpack type to int64");
        }
    }

    arrow::Status convert_to_float(const msgpack_object* obj, arrow::FloatBuilder* builder) {
        if (obj->type == MSGPACK_OBJECT_FLOAT32 || obj->type == MSGPACK_OBJECT_FLOAT64) {
            return builder->Append(static_cast<float>(obj->via.f64));
        } else if (obj->type == MSGPACK_OBJECT_POSITIVE_INTEGER) {
            return builder->Append(static_cast<float>(obj->via.u64));
        } else if (obj->type == MSGPACK_OBJECT_NEGATIVE_INTEGER) {
            return builder->Append(static_cast<float>(obj->via.i64));
        } else {
            /* Return error instead of NULL for type mismatch */
            return arrow::Status::Invalid("Cannot convert msgpack type to float");
        }
    }

    arrow::Status convert_to_double(const msgpack_object* obj, arrow::DoubleBuilder* builder) {
        if (obj->type == MSGPACK_OBJECT_FLOAT32 || obj->type == MSGPACK_OBJECT_FLOAT64) {
            return builder->Append(obj->via.f64);
        } else if (obj->type == MSGPACK_OBJECT_POSITIVE_INTEGER) {
            return builder->Append(static_cast<double>(obj->via.u64));
        } else if (obj->type == MSGPACK_OBJECT_NEGATIVE_INTEGER) {
            return builder->Append(static_cast<double>(obj->via.i64));
        } else if (obj->type == MSGPACK_OBJECT_BOOLEAN) {
            return builder->Append(obj->via.boolean ? 1.0 : 0.0);
        } else {
            /* Return error instead of NULL for type mismatch */
            return arrow::Status::Invalid("Cannot convert msgpack type to double");
        }
    }

    arrow::Status convert_to_string(const msgpack_object* obj, arrow::StringBuilder* builder) {
        if (obj->type == MSGPACK_OBJECT_STR) {
            return builder->Append(obj->via.str.ptr, obj->via.str.size);
        } else if (obj->type == MSGPACK_OBJECT_BIN) {
            return builder->Append(obj->via.bin.ptr, obj->via.bin.size);
        } else if (obj->type == MSGPACK_OBJECT_BOOLEAN) {
            return builder->Append(obj->via.boolean ? "true" : "false");
        } else if (obj->type == MSGPACK_OBJECT_POSITIVE_INTEGER) {
            return builder->Append(std::to_string(obj->via.u64));
        } else if (obj->type == MSGPACK_OBJECT_NEGATIVE_INTEGER) {
            return builder->Append(std::to_string(obj->via.i64));
        } else if (obj->type == MSGPACK_OBJECT_FLOAT32 || obj->type == MSGPACK_OBJECT_FLOAT64) {
            return builder->Append(std::to_string(obj->via.f64));
        } else {
            /* Return error instead of NULL for type mismatch - consistent with other converters */
            return arrow::Status::Invalid("Cannot convert msgpack type to string");
        }
    }

    arrow::Status convert_to_binary(const msgpack_object* obj, arrow::BinaryBuilder* builder) {
        if (obj->type == MSGPACK_OBJECT_BIN) {
            return builder->Append(reinterpret_cast<const uint8_t*>(obj->via.bin.ptr),
                                  obj->via.bin.size);
        } else if (obj->type == MSGPACK_OBJECT_STR) {
            return builder->Append(reinterpret_cast<const uint8_t*>(obj->via.str.ptr),
                                  obj->via.str.size);
        } else {
            /* Return error instead of NULL for type mismatch */
            return arrow::Status::Invalid("Cannot convert msgpack type to binary");
        }
    }

    arrow::Status convert_to_timestamp(const msgpack_object* obj, arrow::TimestampBuilder* builder) {
        /* Get the target time unit from the builder's type */
        auto timestamp_type = std::static_pointer_cast<arrow::TimestampType>(builder->type());
        arrow::TimeUnit::type unit = timestamp_type->unit();

        /* Determine multiplier based on target unit (assume input is in seconds) */
        int64_t multiplier;
        switch (unit) {
            case arrow::TimeUnit::SECOND:
                multiplier = 1;
                break;
            case arrow::TimeUnit::MILLI:
                multiplier = 1000;
                break;
            case arrow::TimeUnit::MICRO:
                multiplier = 1000000;
                break;
            case arrow::TimeUnit::NANO:
                multiplier = 1000000000;
                break;
            default:
                multiplier = 1000000;  /* Default to microseconds */
                break;
        }

        if (obj->type == MSGPACK_OBJECT_POSITIVE_INTEGER) {
            /* Convert Unix timestamp from seconds to target unit */
            return builder->Append(static_cast<int64_t>(obj->via.u64) * multiplier);
        } else if (obj->type == MSGPACK_OBJECT_NEGATIVE_INTEGER) {
            return builder->Append(obj->via.i64 * multiplier);
        } else if (obj->type == MSGPACK_OBJECT_FLOAT32 || obj->type == MSGPACK_OBJECT_FLOAT64) {
            /* Convert Unix timestamp with fractional seconds to target unit */
            return builder->Append(static_cast<int64_t>(obj->via.f64 * multiplier));
        } else {
            /* Return error instead of NULL for type mismatch */
            return arrow::Status::Invalid("Cannot convert msgpack type to timestamp");
        }
    }
};

// Helper function to append type-specific default values for non-nullable fields
bool append_default_value(arrow::ArrayBuilder* builder, const std::shared_ptr<arrow::DataType>& type) {
    switch (type->id()) {
        case arrow::Type::BOOL:
            return static_cast<arrow::BooleanBuilder*>(builder)->Append(false).ok();

        case arrow::Type::INT32:
            return static_cast<arrow::Int32Builder*>(builder)->Append(0).ok();

        case arrow::Type::INT64:
            return static_cast<arrow::Int64Builder*>(builder)->Append(0).ok();

        case arrow::Type::FLOAT:
            return static_cast<arrow::FloatBuilder*>(builder)->Append(0.0f).ok();

        case arrow::Type::DOUBLE:
            return static_cast<arrow::DoubleBuilder*>(builder)->Append(0.0).ok();

        case arrow::Type::STRING:
            return static_cast<arrow::StringBuilder*>(builder)->Append("").ok();

        case arrow::Type::BINARY:
            return static_cast<arrow::BinaryBuilder*>(builder)->Append(static_cast<const uint8_t*>(nullptr), 0).ok();

        case arrow::Type::TIMESTAMP:
            return static_cast<arrow::TimestampBuilder*>(builder)->Append(0).ok();

        default:
            return false;
    }
}

} // anonymous namespace

extern "C" {

void *flb_msgpack_raw_to_parquet(const void *in_buf, size_t in_size,
                                  const char *schema_str,
                                  int compression,
                                  size_t *out_size)
{
    msgpack_unpacked result;
    parquet::Compression::type parquet_compression;
    void *output_buffer = NULL;

    flb_debug("[parquet] Starting msgpack to parquet conversion with user-defined schema\n");

    if (!in_buf || !out_size || !schema_str) {
        flb_error("[parquet] NULL parameter\n");
        return NULL;
    }

    /* Map Fluent Bit compression type to Parquet compression type */
    switch (compression) {
        case 0: /* FLB_AWS_COMPRESS_NONE */
            parquet_compression = parquet::Compression::UNCOMPRESSED;
            break;
        case 1: /* FLB_AWS_COMPRESS_GZIP */
            parquet_compression = parquet::Compression::GZIP;
            break;
        case 2: /* FLB_AWS_COMPRESS_SNAPPY */
            parquet_compression = parquet::Compression::SNAPPY;
            break;
        case 3: /* FLB_AWS_COMPRESS_ZSTD */
            parquet_compression = parquet::Compression::ZSTD;
            break;
        default:
            parquet_compression = parquet::Compression::UNCOMPRESSED;
            flb_warn("[parquet] Unknown compression type %d, defaulting to UNCOMPRESSED\n", compression);
            break;
    }

    try {
        /* 1. Parse user-provided JSON schema first - needed before processing records */
        cJSON* root = cJSON_Parse(schema_str);
        if (!root) {
            const char* error_ptr = cJSON_GetErrorPtr();
            flb_error("[parquet] Failed to parse JSON schema. Schema content: '%s'\n",
                     schema_str);
            if (error_ptr != NULL) {
                flb_error("[parquet] JSON parse error before: '%.20s'\n", error_ptr);
            }
            return NULL;
        }

        cJSON* fields_array = cJSON_GetObjectItem(root, "fields");
        if (!fields_array || !cJSON_IsArray(fields_array)) {
            flb_error("[parquet] Schema must contain 'fields' array. "
                     "Received schema: '%s'\n", schema_str);
            cJSON_Delete(root);
            return NULL;
        }

        /* Build Arrow schema from JSON */
        std::vector<std::shared_ptr<arrow::Field>> arrow_fields;
        int num_fields = cJSON_GetArraySize(fields_array);

        for (int i = 0; i < num_fields; i++) {
            cJSON* field_obj = cJSON_GetArrayItem(fields_array, i);
            if (!field_obj) continue;

            cJSON* name_obj = cJSON_GetObjectItem(field_obj, "name");
            cJSON* type_obj = cJSON_GetObjectItem(field_obj, "type");
            cJSON* nullable_obj = cJSON_GetObjectItem(field_obj, "nullable");

            if (!name_obj || !cJSON_IsString(name_obj)) {
                flb_error("[parquet] Field %d must have 'name' string. "
                         "Schema: '%s'\n", i, schema_str);
                cJSON_Delete(root);
                return NULL;
            }

            if (!type_obj) {
                flb_error("[parquet] Field '%s' missing required 'type' attribute. "
                         "Schema: '%s'\n", name_obj->valuestring, schema_str);
                cJSON_Delete(root);
                return NULL;
            }

            std::string field_name = name_obj->valuestring;
            bool nullable = nullable_obj ? cJSON_IsTrue(nullable_obj) : true;
            std::shared_ptr<arrow::DataType> data_type;

            /* Parse type - support both string and object format */
            if (cJSON_IsString(type_obj)) {
                std::string type_str = type_obj->valuestring;
                if (type_str == "bool" || type_str == "boolean") {
                    data_type = arrow::boolean();
                } else if (type_str == "int32") {
                    data_type = arrow::int32();
                } else if (type_str == "int64") {
                    data_type = arrow::int64();
                } else if (type_str == "float" || type_str == "float32") {
                    data_type = arrow::float32();
                } else if (type_str == "double" || type_str == "float64") {
                    data_type = arrow::float64();
                } else if (type_str == "utf8" || type_str == "string") {
                    data_type = arrow::utf8();
                } else if (type_str == "binary") {
                    data_type = arrow::binary();
                } else {
                    flb_warn("[parquet] Unsupported type '%s', defaulting to utf8\n", type_str.c_str());
                    data_type = arrow::utf8();
                }
            } else if (cJSON_IsObject(type_obj)) {
                cJSON* type_name_obj = cJSON_GetObjectItem(type_obj, "name");
                if (type_name_obj && cJSON_IsString(type_name_obj)) {
                    std::string type_name = type_name_obj->valuestring;
                    if (type_name == "utf8" || type_name == "string") {
                        data_type = arrow::utf8();
                    } else if (type_name == "timestamp") {
                        cJSON* unit_obj = cJSON_GetObjectItem(type_obj, "unit");
                        arrow::TimeUnit::type time_unit = arrow::TimeUnit::MICRO;
                        if (unit_obj && cJSON_IsString(unit_obj)) {
                            std::string unit = unit_obj->valuestring;
                            if (unit == "s") time_unit = arrow::TimeUnit::SECOND;
                            else if (unit == "ms") time_unit = arrow::TimeUnit::MILLI;
                            else if (unit == "us") time_unit = arrow::TimeUnit::MICRO;
                            else if (unit == "ns") time_unit = arrow::TimeUnit::NANO;
                        }
                        data_type = arrow::timestamp(time_unit);
                    } else {
                        flb_warn("[parquet] Unsupported complex type '%s', defaulting to utf8\n", type_name.c_str());
                        data_type = arrow::utf8();
                    }
                } else {
                    data_type = arrow::utf8();
                }
            } else {
                data_type = arrow::utf8();
            }

            arrow_fields.push_back(arrow::field(field_name, data_type, nullable));
        }

        cJSON_Delete(root);

        if (arrow_fields.empty()) {
            flb_error("[parquet] No valid fields found in schema\n");
            return NULL;
        }

        auto schema = arrow::schema(arrow_fields);
        flb_debug("[parquet] Parsed schema with %d fields\n", schema->num_fields());

        /* 2. Create Arrow builders for each field */
        std::vector<std::shared_ptr<arrow::ArrayBuilder>> field_builders;
        for (int i = 0; i < schema->num_fields(); i++) {
            auto field = schema->field(i);
            auto builder_result = arrow::MakeBuilder(field->type(), arrow::default_memory_pool());
            if (!builder_result.ok()) {
                flb_error("[parquet] Failed to create builder for field %s: %s\n",
                         field->name().c_str(), builder_result.status().ToString().c_str());
                return NULL;
            }
            field_builders.push_back(std::move(builder_result).ValueOrDie());
        }

        /* 3. Process records immediately as we unpack - avoid storing msgpack_object pointers */
        MsgpackToArrowConverter converter;

        /* Pre-allocate field lookup map to avoid repeated allocations */
        std::unordered_map<std::string_view, const msgpack_object*> field_map;
        field_map.reserve(schema->num_fields() * 2);

        /* Statistics for missing/conversion failures */
        std::unordered_map<std::string, size_t> missing_field_count;
        std::unordered_map<std::string, size_t> conversion_failure_count;

        /* Helper lambda to process a single map record immediately */
        auto process_record = [&](const msgpack_object* record) -> bool {

            if (record->type != MSGPACK_OBJECT_MAP) {
                /* Append nulls for all fields */
                for (auto& builder : field_builders) {
                    auto status = builder->AppendNull();
                    if (!status.ok()) {
                        flb_error("[parquet] Failed to append null for non-map record: %s\n",
                                 status.ToString().c_str());
                        return false;
                    }
                }
                return true;
            }

            auto& map = record->via.map;

            /* Build field lookup map using string_view to avoid string copies */
            field_map.clear();  /* Reuse the map instead of creating new one */
            for (uint32_t i = 0; i < map.size; i++) {
                auto& kv = map.ptr[i];
                if (kv.key.type == MSGPACK_OBJECT_STR) {
                    /* Use string_view to avoid string allocation/copy */
                    std::string_view key(kv.key.via.str.ptr, kv.key.via.str.size);
                    field_map[key] = &kv.val;
                }
            }

            /* Append values for each field */
            for (int field_idx = 0; field_idx < schema->num_fields(); field_idx++) {
                auto field = schema->field(field_idx);
                auto& builder = field_builders[field_idx];

                auto it = field_map.find(field->name());
                if (it != field_map.end()) {
                    auto status = converter.convert_value(it->second, builder.get(), field->type(), field->name());
                    if (!status.ok()) {
                        /* Conversion failed - check if field is nullable */
                        if (field->nullable()) {
                            auto null_status = builder->AppendNull();
                            if (!null_status.ok()) {
                                flb_error("[parquet] Failed to append null after conversion failure: %s\n",
                                         null_status.ToString().c_str());
                                return false;
                            }
                        } else {
                            /* Non-nullable field - append default value and track statistics */
                            conversion_failure_count[field->name()]++;
                            if (!append_default_value(builder.get(), field->type())) {
                                flb_error("[parquet] Failed to append default value for field '%s'\n",
                                         field->name().c_str());
                                return false;
                            }
                        }
                    }
                } else {
                    /* Field is missing from data */
                    if (field->nullable()) {
                        /* Nullable field - append null */
                        auto null_status = builder->AppendNull();
                        if (!null_status.ok()) {
                            flb_error("[parquet] Failed to append null for missing field '%s': %s\n",
                                     field->name().c_str(), null_status.ToString().c_str());
                            return false;
                        }
                    } else {
                        /* Non-nullable field - append default value and track statistics */
                        missing_field_count[field->name()]++;
                        if (!append_default_value(builder.get(), field->type())) {
                            flb_error("[parquet] Failed to append default value for field '%s'\n",
                                     field->name().c_str());
                            return false;
                        }
                    }
                }
            }
            return true;
        };

        /* Unpack and process records immediately */
        msgpack_unpacked_init(&result);
        size_t offset = 0;
        size_t record_count = 0;

        while (msgpack_unpack_next(&result, static_cast<const char*>(in_buf), in_size, &offset)
               == MSGPACK_UNPACK_SUCCESS) {
            if (result.data.type == MSGPACK_OBJECT_ARRAY) {
                /* Process each element in the array */
                auto& array = result.data.via.array;
                for (uint32_t i = 0; i < array.size; i++) {
                    if (!process_record(&array.ptr[i])) {
                        msgpack_unpacked_destroy(&result);
                        return NULL;
                    }
                    record_count++;
                }
            } else {
                /* Process single record */
                if (!process_record(&result.data)) {
                    msgpack_unpacked_destroy(&result);
                    return NULL;
                }
                record_count++;
            }
        }

        msgpack_unpacked_destroy(&result);

        if (record_count == 0) {
            flb_error("[parquet] No valid records found in msgpack data\n");
            return NULL;
        }

        flb_debug("[parquet] Processed %zu records\n", record_count);

        /* Output statistics summary - only if there are data quality issues */
        bool has_issues = !missing_field_count.empty() || !conversion_failure_count.empty() ||
                         !converter.int32_overflow_by_field.empty() || !converter.int64_overflow_by_field.empty() ||
                         !converter.float_to_int_clamp_by_field.empty();

        if (has_issues) {
            flb_warn("[parquet] Data quality summary for %zu records:", record_count);

            if (!converter.int32_overflow_by_field.empty()) {
                flb_warn("[parquet] Integer overflow (int64/uint64 -> int32 clamped):");
                for (const auto& pair : converter.int32_overflow_by_field) {
                    flb_warn("[parquet]   field='%s' count=%zu",
                            pair.first.c_str(), pair.second);
                }
            }

            if (!converter.int64_overflow_by_field.empty()) {
                flb_warn("[parquet] Integer overflow (uint64 -> int64 clamped):");
                for (const auto& pair : converter.int64_overflow_by_field) {
                    flb_warn("[parquet]   field='%s' count=%zu",
                            pair.first.c_str(), pair.second);
                }
            }

            if (!converter.float_to_int_clamp_by_field.empty()) {
                flb_warn("[parquet] Float to int32 clamping:");
                for (const auto& pair : converter.float_to_int_clamp_by_field) {
                    flb_warn("[parquet]   field='%s' count=%zu",
                            pair.first.c_str(), pair.second);
                }
            }

            if (!missing_field_count.empty()) {
                flb_warn("[parquet] Missing non-nullable fields (defaults used):");
                for (const auto& pair : missing_field_count) {
                    flb_warn("[parquet]   field='%s' count=%zu",
                            pair.first.c_str(), pair.second);
                }
            }

            if (!conversion_failure_count.empty()) {
                flb_warn("[parquet] Type conversion failures (defaults used):");
                for (const auto& pair : conversion_failure_count) {
                    flb_warn("[parquet]   field='%s' count=%zu",
                            pair.first.c_str(), pair.second);
                }
            }
        }

        /* 4. Finish building arrays */
        std::vector<std::shared_ptr<arrow::Array>> arrays;
        for (size_t i = 0; i < field_builders.size(); i++) {
            auto array_result = field_builders[i]->Finish();
            if (!array_result.ok()) {
                flb_error("[parquet] Failed to finish array for field %d: %s\n",
                         static_cast<int>(i), array_result.status().ToString().c_str());
                return NULL;
            }
            arrays.push_back(array_result.ValueOrDie());
        }

        /* 5. Create RecordBatch */
        auto batch = arrow::RecordBatch::Make(schema, record_count, arrays);
        flb_debug("[parquet] Created RecordBatch with %lld rows and %d columns\n",
                 batch->num_rows(), batch->num_columns());

        /* 6. Let Arrow manage the buffer - it will automatically grow as needed */
        auto output_stream_result = arrow::io::BufferOutputStream::Create(
            INITIAL_PARQUET_BUFFER_SIZE,  /* Initial size hint, Arrow will auto-grow */
            arrow::default_memory_pool());

        if (!output_stream_result.ok()) {
            flb_error("[parquet] Failed to create output stream: %s\n",
                     output_stream_result.status().ToString().c_str());
            return NULL;
        }
        auto output_stream = output_stream_result.ValueOrDie();

        /* 7. Create Parquet writer properties */
        auto writer_properties = parquet::WriterProperties::Builder()
            .compression(parquet_compression)
            ->build();

        auto arrow_writer_properties = parquet::ArrowWriterProperties::Builder().build();

        /* 8. Write to Parquet - Arrow automatically manages memory growth */
        auto writer_result = parquet::arrow::FileWriter::Open(
            *schema.get(),
            arrow::default_memory_pool(),
            output_stream,
            writer_properties,
            arrow_writer_properties);

        if (!writer_result.ok()) {
            flb_error("[parquet] Failed to create Parquet writer: %s\n",
                     writer_result.status().ToString().c_str());
            return NULL;
        }
        std::unique_ptr<parquet::arrow::FileWriter> writer = std::move(writer_result).ValueOrDie();

        auto status = writer->WriteRecordBatch(*batch);
        if (!status.ok()) {
            flb_error("[parquet] Failed to write RecordBatch: %s\n",
                     status.ToString().c_str());
            return NULL;
        }

        status = writer->Close();
        if (!status.ok()) {
            flb_error("[parquet] Failed to close writer: %s\n",
                     status.ToString().c_str());
            return NULL;
        }

        /* 9. Get the final buffer from Arrow - it knows the exact size */
        auto buffer_result = output_stream->Finish();
        if (!buffer_result.ok()) {
            flb_error("[parquet] Failed to finish output stream: %s\n",
                     buffer_result.status().ToString().c_str());
            return NULL;
        }
        auto arrow_buffer = buffer_result.ValueOrDie();

        /* 10. Copy to Fluent Bit managed memory */
        size_t final_size = arrow_buffer->size();
        output_buffer = flb_malloc(final_size);
        if (!output_buffer) {
            flb_error("[parquet] Failed to allocate %zu bytes for output\n", final_size);
            return NULL;
        }

        memcpy(output_buffer, arrow_buffer->data(), final_size);
        *out_size = final_size;

        flb_debug("[parquet] Successfully converted to Parquet: %zu bytes\n", final_size);

        return output_buffer;

    } catch (const parquet::ParquetException& e) {
        flb_error("[parquet] Parquet exception: %s\n", e.what());
        if (output_buffer) {
            flb_free(output_buffer);
        }
        return NULL;
    } catch (const std::exception& e) {
        flb_error("[parquet] Exception during conversion: %s\n", e.what());
        if (output_buffer) {
            flb_free(output_buffer);
        }
        return NULL;
    }
}

} // extern "C"
