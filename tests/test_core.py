import unittest

import pandas as pd

from src.entities import ItemResolver, SupplierResolver, labels_by_line
from src.matching import match_tool_records
from src.utils import extract_po_and_item, parse_number, parse_vat_rate, stable_key


class UtilityTests(unittest.TestCase):
    def test_parse_us_finance_amount(self):
        self.assertAlmostEqual(parse_number("5,266.63"), 5266.63)

    def test_parse_vat_as_fraction(self):
        self.assertAlmostEqual(parse_vat_rate("15%"), 0.15)
        self.assertAlmostEqual(parse_vat_rate("0.15"), 0.15)

    def test_extract_po_and_item(self):
        self.assertEqual(extract_po_and_item("PO 4510109330/10.0"), ("4510109330", "10"))

    def test_stable_key_is_deterministic(self):
        self.assertEqual(stable_key("SG", "Example Supplier"), stable_key("SG", "Example Supplier"))


class EntityTests(unittest.TestCase):
    def test_labels_and_golden_entity_keys(self):
        labels = pd.DataFrame([{
            "record_id": "ERP:A000001", "category_l1": "L1", "category_l2": "L2",
            "category_l3": "L3", "supplier_key": "SG-golden", "item_key": "IG-golden",
        }])
        mapped = labels_by_line(labels)
        self.assertIn("A000001", mapped)

        erp = pd.DataFrame([{
            "line_id": "A000001", "supplier_name": "Example Supplier (Pty) Ltd",
            "vendor_id": "V1", "material_number": "M1", "material_short_text": "Steel bolt",
            "service_short_text": "", "material_type": "Material",
        }])
        supplier = SupplierResolver(erp, mapped)
        item = ItemResolver(erp, mapped)
        self.assertEqual(supplier.key_for_erp_index(0), "SG-golden")
        self.assertEqual(item.key_for_erp_index(0), "IG-golden")


class MatchingTests(unittest.TestCase):
    def test_one_to_one_po_match(self):
        erp = pd.DataFrame([{
            "po_number": "4510109330", "po_item": 10, "supplier_name": "Example Supplier Pty Ltd",
            "material_short_text": "Steel bolt", "service_short_text": "", "posting_date": "2024-01-01",
        }])
        invoice = pd.DataFrame([{
            "po_reference": "PO 4510109330/10.0", "supplier_name_as_captured": "Example Supplier",
            "line_text": "Steel bolt", "invoice_date": "2024-01-02",
        }])
        result = match_tool_records(erp, invoice, [100.0], [100.0], {
            "po_match_threshold": 0.70, "supplier_name_threshold": 0.90,
        })
        self.assertIn(0, result)
        self.assertEqual(result[0][0], 0)


if __name__ == "__main__":
    unittest.main()
