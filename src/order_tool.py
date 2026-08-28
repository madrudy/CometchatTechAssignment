import json
import re
from pathlib import Path
from .models import ToolResult

ORDER_RE = re.compile(r"\bORD-\d{4}\b", re.IGNORECASE)

class OrderTool:
    
    def __init__(self, orders_path: str | Path):
        raw = json.loads(Path(orders_path).read_text(encoding="utf-8"))
        self.orders = {o["order_id"].upper(): o for o in raw["orders"]}

    @staticmethod
    def extract_order_id(text: str) -> str | None:
        match = ORDER_RE.search(text or "")
        return match.group(0).upper() if match else None

    def lookup(self, order_id: str) -> ToolResult:
        normalized = (order_id or "").strip().upper()
        if not re.fullmatch(r"ORD-\d{4}", normalized):
            return ToolResult("malformed", {"message": "The order ID format is invalid."})
        order = self.orders.get(normalized)
        if not order:
            return ToolResult("not_found", {"message": "Order was not found. Check the order ID or contact support."})

        status = order["status"].lower()
        data = {
            "order_id": order["order_id"],
            "status": status,
            "customer_safe_message": order["customer_safe_message"],
        }
        if status not in {"cancelled", "returned"}:
            if order.get("carrier"):
                data["carrier"] = order["carrier"]
            if order.get("tracking_number"):
                data["tracking_number"] = order["tracking_number"]
            if order.get("estimated_delivery"):
                data["estimated_delivery"] = order["estimated_delivery"]

        return ToolResult("found", data)
