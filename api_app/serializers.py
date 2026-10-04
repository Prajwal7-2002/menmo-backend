from rest_framework import serializers

from .models import Feedback, QueryLog


class AskSerializer(serializers.Serializer):
    query = serializers.CharField(max_length=4000)


class FeedbackSerializer(serializers.ModelSerializer):
    query_id = serializers.UUIDField(write_only=True)

    class Meta:
        model = Feedback
        fields = ['id', 'query_id', 'value', 'reason', 'created_at']
        read_only_fields = ['id', 'created_at']

    def validate_query_id(self, value):
        # Only the user who asked may rate an answer (ratings shift chunk ranking).
        user = self.context["request"].user
        if not QueryLog.objects.filter(id=value, user=user).exists():
            raise serializers.ValidationError("query_id not found")
        return value

    def create(self, validated_data):
        qlog = QueryLog.objects.get(id=validated_data.pop("query_id"))
        return Feedback.objects.create(query_log=qlog, user=self.context["request"].user, **validated_data)
