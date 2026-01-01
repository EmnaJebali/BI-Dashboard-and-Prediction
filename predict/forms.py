from django import forms


class SatisfactionForm(forms.Form):
    gender = forms.ChoiceField(
        choices=[('Male', 'Male'), ('Female', 'Female')],
        label="Gender",
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    customer_type = forms.ChoiceField(
        choices=[('Loyal Customer', 'Loyal Customer'), ('disloyal Customer', 'Disloyal Customer')],
        label="Customer Type",
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    age = forms.IntegerField(
        min_value=7, 
        max_value=85, 
        label="Age",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter your age'})
    )
    type_of_travel = forms.ChoiceField(
        choices=[('Personal Travel', 'Personal'), ('Business travel', 'Business')],
        label="Type of Travel",
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    flight_class = forms.ChoiceField(
        choices=[('Eco', 'Economy'), ('Eco Plus', 'Economy Plus'), ('Business', 'Business')],
        label="Class",
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    flight_distance = forms.IntegerField(
        min_value=31, 
        max_value=4983, 
        label="Flight Distance (miles)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Distance in miles'})
    )

    seat_comfort = forms.IntegerField(
        min_value=0, 
        max_value=5, 
        initial=3, 
        label="Seat Comfort",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'type': 'range', 'min': '0', 'max': '5'})
    )
    departure_arrival_time = forms.IntegerField(
        min_value=0, 
        max_value=5, 
        initial=3, 
        label="Departure/Arrival Time Convenience",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'type': 'range', 'min': '0', 'max': '5'})
    )
    food_and_drink = forms.IntegerField(
        min_value=0, 
        max_value=5, 
        initial=3, 
        label="Food and Drink",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'type': 'range', 'min': '0', 'max': '5'})
    )
    gate_location = forms.IntegerField(
        min_value=0, 
        max_value=5, 
        initial=3, 
        label="Gate Location",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'type': 'range', 'min': '0', 'max': '5'})
    )
    inflight_wifi = forms.IntegerField(
        min_value=0, 
        max_value=5, 
        initial=3, 
        label="Inflight Wifi Service",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'type': 'range', 'min': '0', 'max': '5'})
    )
    inflight_entertainment = forms.IntegerField(
        min_value=0, 
        max_value=5, 
        initial=3, 
        label="Inflight Entertainment",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'type': 'range', 'min': '0', 'max': '5'})
    )
    online_support = forms.IntegerField(
        min_value=0, 
        max_value=5, 
        initial=3, 
        label="Online Support",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'type': 'range', 'min': '0', 'max': '5'})
    )
    ease_of_online_booking = forms.IntegerField(
        min_value=0, 
        max_value=5, 
        initial=3, 
        label="Ease of Online Booking",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'type': 'range', 'min': '0', 'max': '5'})
    )
    on_board_service = forms.IntegerField(
        min_value=0, 
        max_value=5, 
        initial=3, 
        label="On-board Service",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'type': 'range', 'min': '0', 'max': '5'})
    )
    leg_room_service = forms.IntegerField(
        min_value=0, 
        max_value=5, 
        initial=3, 
        label="Leg Room Service",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'type': 'range', 'min': '0', 'max': '5'})
    )
    baggage_handling = forms.IntegerField(
        min_value=0, 
        max_value=5, 
        initial=3, 
        label="Baggage Handling",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'type': 'range', 'min': '0', 'max': '5'})
    )
    checkin_service = forms.IntegerField(
        min_value=0, 
        max_value=5, 
        initial=3, 
        label="Check-in Service",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'type': 'range', 'min': '0', 'max': '5'})
    )
    cleanliness = forms.IntegerField(
        min_value=0, 
        max_value=5, 
        initial=3, 
        label="Cleanliness",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'type': 'range', 'min': '0', 'max': '5'})
    )
    online_boarding = forms.IntegerField(
        min_value=0, 
        max_value=5, 
        initial=3, 
        label="Online Boarding",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'type': 'range', 'min': '0', 'max': '5'})
    )

    departure_delay = forms.IntegerField(
        min_value=0, 
        initial=0, 
        required=False, 
        label="Departure Delay (minutes)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '0'})
    )
    arrival_delay = forms.IntegerField(
        min_value=0, 
        initial=0, 
        required=False, 
        label="Arrival Delay (minutes)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '0'})
    )


class ClusteringPartnersForm(forms.Form):
    total_issued_points = forms.IntegerField(
        min_value=0,
        label="Total Issued Points",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter total loyalty points issued'})
    )
    total_cost = forms.IntegerField(
        min_value=0,
        label="Total Cost",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter total cost of issued points'})
    )

class CLVForm(forms.Form):
    salary = forms.IntegerField(
        min_value=0,
        label="Annual Salary ($)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter annual salary'})
    )
    tier_rank = forms.IntegerField(
        min_value=1,
        max_value=3,
        label="Loyalty Tier Rank (1-3)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter tier rank (1=lowest, 3=highest)'})
    )
    tenure_months = forms.IntegerField(
        min_value=0,
        label="Tenure (months)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'How many months as customer'})
    )
    total_flights = forms.IntegerField(
        min_value=0,
        label="Total Flights",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Total number of flights'})
    )
    distance = forms.IntegerField(
        min_value=0,
        label="Total Distance Flown (miles)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Total distance in miles'})
    )
    points_accumulated = forms.IntegerField(
        min_value=0,
        label="Points Accumulated",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Total loyalty points earned'})
    )
    points_redeemed = forms.IntegerField(
        min_value=0,
        label="Points Redeemed",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Total loyalty points used'})
    )
    current_clv = forms.FloatField(
        min_value=0,
        label="Current/Historical CLV ($)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter current customer lifetime value'})
    )