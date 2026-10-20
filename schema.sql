-- Main customer dataset: one row per customer from the Telco Customer Churn dataset.
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    gender TEXT,
    senior_citizen INTEGER CHECK (senior_citizen IN (0,1)),
    partner TEXT,
    dependents TEXT,
    tenure_months INTEGER,
    phone_service TEXT,
    multiple_lines TEXT,
    internet_service TEXT,
    online_security TEXT,
    online_backup TEXT,
    device_protection TEXT,
    tech_support TEXT,
    streaming_tv TEXT,
    streaming_movies TEXT,
    contract TEXT,
    paperless_billing TEXT,
    payment_method TEXT,
    monthly_charges REAL,
    total_charges REAL,
    churn TEXT CHECK (churn IN ('Yes','No'))
);

CREATE INDEX IF NOT EXISTS idx_customers_churn ON customers(churn);
CREATE INDEX IF NOT EXISTS idx_customers_contract ON customers(contract);

-- Cleaned, feature-engineered table derived from `customers`, built by clean_data.py.
-- Ready for model training: no nulls, binary/one-hot encoded categoricals.
-- `customers` itself is never modified; this table can be rebuilt at any time.
CREATE TABLE IF NOT EXISTS customers_ml (
    customer_id TEXT PRIMARY KEY REFERENCES customers(customer_id),
    tenure_months INTEGER NOT NULL,
    tenure_group TEXT NOT NULL,
    monthly_charges REAL NOT NULL,
    total_charges REAL NOT NULL,
    total_charges_was_imputed INTEGER NOT NULL CHECK (total_charges_was_imputed IN (0,1)),
    avg_monthly_spend REAL NOT NULL,
    senior_citizen INTEGER NOT NULL CHECK (senior_citizen IN (0,1)),
    gender_male INTEGER NOT NULL CHECK (gender_male IN (0,1)),
    has_partner INTEGER NOT NULL CHECK (has_partner IN (0,1)),
    has_dependents INTEGER NOT NULL CHECK (has_dependents IN (0,1)),
    has_phone_service INTEGER NOT NULL CHECK (has_phone_service IN (0,1)),
    has_paperless_billing INTEGER NOT NULL CHECK (has_paperless_billing IN (0,1)),
    churn INTEGER NOT NULL CHECK (churn IN (0,1)),
    multiple_lines_no INTEGER NOT NULL CHECK (multiple_lines_no IN (0,1)),
    multiple_lines_yes INTEGER NOT NULL CHECK (multiple_lines_yes IN (0,1)),
    multiple_lines_no_phone_service INTEGER NOT NULL CHECK (multiple_lines_no_phone_service IN (0,1)),
    internet_service_dsl INTEGER NOT NULL CHECK (internet_service_dsl IN (0,1)),
    internet_service_fiber_optic INTEGER NOT NULL CHECK (internet_service_fiber_optic IN (0,1)),
    internet_service_no INTEGER NOT NULL CHECK (internet_service_no IN (0,1)),
    online_security_no INTEGER NOT NULL CHECK (online_security_no IN (0,1)),
    online_security_yes INTEGER NOT NULL CHECK (online_security_yes IN (0,1)),
    online_security_no_internet_service INTEGER NOT NULL CHECK (online_security_no_internet_service IN (0,1)),
    online_backup_no INTEGER NOT NULL CHECK (online_backup_no IN (0,1)),
    online_backup_yes INTEGER NOT NULL CHECK (online_backup_yes IN (0,1)),
    online_backup_no_internet_service INTEGER NOT NULL CHECK (online_backup_no_internet_service IN (0,1)),
    device_protection_no INTEGER NOT NULL CHECK (device_protection_no IN (0,1)),
    device_protection_yes INTEGER NOT NULL CHECK (device_protection_yes IN (0,1)),
    device_protection_no_internet_service INTEGER NOT NULL CHECK (device_protection_no_internet_service IN (0,1)),
    tech_support_no INTEGER NOT NULL CHECK (tech_support_no IN (0,1)),
    tech_support_yes INTEGER NOT NULL CHECK (tech_support_yes IN (0,1)),
    tech_support_no_internet_service INTEGER NOT NULL CHECK (tech_support_no_internet_service IN (0,1)),
    streaming_tv_no INTEGER NOT NULL CHECK (streaming_tv_no IN (0,1)),
    streaming_tv_yes INTEGER NOT NULL CHECK (streaming_tv_yes IN (0,1)),
    streaming_tv_no_internet_service INTEGER NOT NULL CHECK (streaming_tv_no_internet_service IN (0,1)),
    streaming_movies_no INTEGER NOT NULL CHECK (streaming_movies_no IN (0,1)),
    streaming_movies_yes INTEGER NOT NULL CHECK (streaming_movies_yes IN (0,1)),
    streaming_movies_no_internet_service INTEGER NOT NULL CHECK (streaming_movies_no_internet_service IN (0,1)),
    contract_month_to_month INTEGER NOT NULL CHECK (contract_month_to_month IN (0,1)),
    contract_one_year INTEGER NOT NULL CHECK (contract_one_year IN (0,1)),
    contract_two_year INTEGER NOT NULL CHECK (contract_two_year IN (0,1)),
    payment_method_electronic_check INTEGER NOT NULL CHECK (payment_method_electronic_check IN (0,1)),
    payment_method_mailed_check INTEGER NOT NULL CHECK (payment_method_mailed_check IN (0,1)),
    payment_method_bank_transfer_automatic INTEGER NOT NULL CHECK (payment_method_bank_transfer_automatic IN (0,1)),
    payment_method_credit_card_automatic INTEGER NOT NULL CHECK (payment_method_credit_card_automatic IN (0,1))
);

CREATE INDEX IF NOT EXISTS idx_customers_ml_churn ON customers_ml(churn);

-- Each prediction is stored separately so that model history can be evaluated.
CREATE TABLE IF NOT EXISTS churn_predictions (
    prediction_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id TEXT NOT NULL,
    churn_probability REAL NOT NULL CHECK (churn_probability >= 0 AND churn_probability <= 1),
    predicted_churn TEXT NOT NULL CHECK (predicted_churn IN ('Yes','No')),
    model_version TEXT,
    predicted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_customer ON churn_predictions(customer_id);
CREATE INDEX IF NOT EXISTS idx_predictions_time ON churn_predictions(predicted_at);

-- Metadata for provider's official plan comparison / terms-of-service documents.
-- The actual PDF must come from the chosen provider; do not populate with invented terms.
CREATE TABLE IF NOT EXISTS plan_documents (
    document_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    provider_name TEXT NOT NULL,
    document_title TEXT NOT NULL,
    document_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    source_url TEXT,
    version_label TEXT,
    effective_date TEXT,
    ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Minimal conversation audit trail. Avoid storing unnecessary sensitive personal data.
CREATE TABLE IF NOT EXISTS chat_history (
    chat_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id TEXT,
    user_question TEXT NOT NULL,
    assistant_answer TEXT,
    source_document_id INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (source_document_id) REFERENCES plan_documents(document_id)
);
