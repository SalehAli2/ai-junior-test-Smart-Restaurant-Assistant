from langchain_core.tools import tool
from datetime import date
import uuid

@tool
def check_table_availability(date: str, time: str, branch: str) -> str:
    """
    Check if a table is available at a NovaBite branch.
    Args:
        date: Reservation date in YYYY-MM-DD format
        time: Reservation time in HH:MM format
        branch: Branch name (Downtown, Uptown, or Waterfront)
    """
    branch_normalized = branch.lower().strip()
    seed = hash(f"{date}{time}{branch_normalized}") % 100
    is_available = seed > 30

    if is_available:
        return (
            f"Table confirmed available at NovaBite {branch.capitalize()} "
            f"on {date} at {time}."
        )
    else:
        all_times = ["12:00", "13:00", "18:00", "18:30", "21:00", "21:30"]
        suggestions = ", ".join(
            list(t for t in all_times if t != time)[:3]
            )
        return (
            f"NovaBite {branch.capitalize()} is fully booked on {date} at {time}. "
            f"Available times on the same date: {suggestions}. "
            f"Which would you prefer?"
        )


@tool
def book_table(name: str, date: str, time: str, branch: str, party_size: int) -> str:
    """
    Book a table at a NovaBite branch.
    Args:
        name: Guest name
        date: Reservation date in YYYY-MM-DD format
        time: Reservation time in HH:MM format
        branch: Branch name (Downtown, Uptown, or Waterfront)
        party_size: Number of guests
    """
    booking_id = f"NB-2026-{uuid.uuid4().hex[:4].upper()}"
    return (
        f"Booking Confirmed!\n"
        f"Reservation ID: {booking_id}\n"
        f"Guest Name: {name}\n"
        f"Location: NovaBite {branch.capitalize()}\n"
        f"Date: {date}\n"
        f"Time: {time}\n"
        f"Party Size: {party_size} people\n"
        f"We look forward to seeing you!"
    )


@tool
def get_today_special(branch: str) -> str:
    """
    Get today's chef special at a NovaBite branch.
    Args:
        branch: Branch name (Downtown, Uptown, or Waterfront)
    """
    branch_normalized = branch.lower().strip()
    daily_specials = {
        "downtown": [
            "Truffle Mushroom Risotto with shaved Parmigiano Reggiano.",
            "Osso Buco alla Milanese with saffron risotto.",
            "Spaghetti al Nero di Seppia with fresh seafood."
        ],
        "uptown": [
            "Pan-seared Sea Bass with lemon-caper butter sauce.",
            "Bistecca alla Fiorentina with roasted rosemary potatoes.",
            "Burrata e Prosciutto with aged balsamic and truffle honey."
        ],
        "waterfront": [
            "Linguine ai Frutti di Mare with fresh daily catch.",
            "Grilled Branzino with caponata and herb oil.",
            "Prawn and Saffron Risotto with crispy pancetta."
        ]
    }
    day_index = date.today().day % 3
    specials = daily_specials.get(
        branch_normalized,
        ["Classic Margherita Pizza with Buffalo Mozzarella."]
    )
    special = specials[day_index % len(specials)]
    return f"Today's special at NovaBite {branch.capitalize()}: {special}"


@tool
def check_loyalty_points(user_id: str) -> str:
    """
    Check loyalty points balance and tier for a NovaBite member.
    Args:
        user_id: The customer's loyalty account ID
    """
    known_users = {
        "user_123": {"points": 450, "tier": "Silver"},
        "user_456": {"points": 1200, "tier": "Gold"},
        "user_789": {"points": 1800, "tier": "Platinum"}
    }

    if user_id in known_users:
        data = known_users[user_id]
        points = data["points"]
        tier = data["tier"]
    else:
        points = hash(user_id) % 1500
        points = abs(points)
        tier = "Silver" if points < 500 else "Gold" if points < 1500 else "Platinum"

    discount = (points // 100) * 5
    return (
        f"Loyalty Account: {user_id}\n"
        f"Points Balance: {points} points\n"
        f"Current Tier: {tier}\n"
        f"Redeemable Discount: ${discount} USD"
    )